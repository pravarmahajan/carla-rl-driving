"""Create immutable, self-describing directories for experiment runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from .config import ExperimentConfig, ServerCatalog, canonical_fingerprint


DEFAULT_REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_CATALOG = DEFAULT_REPOSITORY / "configs" / "servers.json"
DEFAULT_CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", Path.home() / "git" / "carla"))
PACKAGE_NAMES = ("carla", "gymnasium", "stable-baselines3", "torch", "numpy", "cloudpickle")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_patch(repository: Path) -> str:
    """Capture tracked and untracked changes in a single Git-compatible patch."""
    tracked = _git(repository, "diff", "--binary", "HEAD")
    untracked_result = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    sections = [tracked] if tracked else []
    for raw_path in untracked_result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        result = subprocess.run(
            ("git", "diff", "--no-index", "--binary", "--", "/dev/null", relative_path),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        # git diff --no-index returns 1 when differences were found.
        if result.returncode not in (0, 1):
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        if result.stdout:
            sections.append(result.stdout.rstrip())
    return "\n".join(sections) + ("\n" if sections else "")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _external_carla_source(carla_root: Path) -> dict[str, Any]:
    agents_path = carla_root / "PythonAPI" / "carla" / "agents"
    if not agents_path.exists():
        return {"root": str(carla_root), "available": False}
    try:
        commit = _git(carla_root, "rev-parse", "HEAD")
        status = _git(carla_root, "status", "--short", "--", "PythonAPI/carla/agents")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"root": str(carla_root), "available": True, "git_repository": False}
    return {
        "root": str(carla_root),
        "available": True,
        "git_repository": True,
        "commit": commit,
        "agents_dirty": bool(status),
        "agents_status": status.splitlines(),
    }


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not cleaned:
        raise ValueError("run name must contain at least one letter or digit")
    return cleaned


def create_run(
    experiment: ExperimentConfig,
    catalog: ServerCatalog,
    server_name: str,
    runs_root: Path,
    repository: Path = DEFAULT_REPOSITORY,
    carla_root: Path = DEFAULT_CARLA_ROOT,
    run_name: str | None = None,
    now: datetime | None = None,
) -> Path:
    server = catalog.get(server_name)
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S.%fZ")
    name = _slug(run_name or str(experiment.data["name"]))
    run_id = f"{timestamp}-{name}"
    run_dir = runs_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    git_commit = _git(repository, "rev-parse", "HEAD")
    git_status = _git(repository, "status", "--short")
    git_diff = _git_patch(repository)

    resolved = json.loads(json.dumps(experiment.data))
    resolved["simulator"]["server"] = server.name
    resolved["simulator"]["host"] = server.host
    resolved["simulator"]["rpc_port"] = server.rpc_port
    resolved["simulator"]["traffic_manager_port"] = server.traffic_manager_port

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": timestamp,
        "status": "created",
        "experiment": {
            "name": experiment.data["name"],
            "source": str(experiment.source),
            "fingerprint": experiment.fingerprint,
            "resolved_fingerprint": canonical_fingerprint(resolved),
        },
        "server_catalog": {
            "source": str(catalog.source),
            "fingerprint": catalog.fingerprint,
            "selected": server.name,
        },
        "git": {
            "commit": git_commit,
            "dirty": bool(git_status),
            "status": git_status.splitlines(),
        },
        "packages": _package_versions(),
        "external_sources": {
            "carla_python_agents": _external_carla_source(carla_root),
        },
    }

    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "git.patch").write_text(git_diff, encoding="utf-8")
    return run_dir


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_SERVER_CATALOG)
    parser.add_argument("--server", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_REPOSITORY / "runs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = create_run(
        ExperimentConfig.load(args.config),
        ServerCatalog.load(args.catalog),
        args.server,
        args.runs_root,
        run_name=args.run_name,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
