"""Run a client process while holding an exclusive CARLA server-slot lease."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator, Sequence

from .config import ConfigurationError, ServerCatalog, ServerConfig


DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "configs" / "servers.json"
DEFAULT_LOCK_DIR = Path(os.environ.get("CARLA_SLOT_LOCK_DIR", "/tmp/carla-rl-driving/slot-locks"))


class NoServerSlotAvailable(RuntimeError):
    pass


@contextmanager
def acquire_server(
    catalog: ServerCatalog,
    requested: str = "auto",
    lock_dir: Path = DEFAULT_LOCK_DIR,
) -> Iterator[ServerConfig]:
    """Acquire one process-wide exclusive lease using a non-blocking file lock."""
    candidates = catalog.servers if requested == "auto" else (catalog.get(requested),)
    lock_dir.mkdir(parents=True, exist_ok=True)

    for server in candidates:
        lock_path = lock_dir / f"{server.name}.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue

        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "server": server.name}, handle)
        handle.flush()
        try:
            yield server
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        return

    if requested == "auto":
        raise NoServerSlotAvailable("No CARLA server slot is currently available")
    raise NoServerSlotAvailable(f"CARLA server slot {requested!r} is currently leased")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--slot", default="auto", help="Slot name or 'auto'")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide a client command after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        catalog = ServerCatalog.load(args.catalog)
        with acquire_server(catalog, args.slot) as server:
            environment = os.environ.copy()
            environment.update(server.as_environment())
            print(
                f"Leased {server.name}: {server.host}:{server.rpc_port} "
                f"(traffic manager {server.traffic_manager_port})",
                flush=True,
            )
            return subprocess.run(args.command, env=environment, check=False).returncode
    except (ConfigurationError, NoServerSlotAvailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

