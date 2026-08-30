"""Shared command-line support for config-backed CARLA client runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import ExperimentConfig, ServerCatalog
from .run_manifest import DEFAULT_SERVER_CATALOG, create_run

SUPPORTED_OBSERVATION_FEATURES = (
    "speed_kmh",
    "distance_to_next_waypoint_m",
    "route_heading_error_rad",
    "distance_to_goal_m",
    "lane_offset_m",
    "lane_heading_error_rad",
    "previous_steer",
    "previous_throttle",
    "obstacle_distance_m",
)
SUPPORTED_ACTION_FEATURES = ("steer", "throttle", "brake")


def _validate_current_environment(experiment: ExperimentConfig) -> None:
    """Fail rather than silently claim a config changed unsupported env behavior."""
    environment = experiment.data["environment"]
    if tuple(environment["observation_features"]) != SUPPORTED_OBSERVATION_FEATURES:
        raise ValueError(
            "This code currently implements only the state-v1 observation layout; "
            "refactor the observation feature registry before selecting a different layout."
        )
    if tuple(environment["action_features"]) != SUPPORTED_ACTION_FEATURES:
        raise ValueError("This code currently implements only vehicle-control-v1 actions")
    town = experiment.data["simulator"].get("town")
    if town is not None:
        raise ValueError("This code does not yet load towns from experiment config")


def create_configured_run(
    config_path: Path,
    catalog_path: Path,
    server_name: str | None,
    runs_root: Path,
    run_name: str | None,
    command: str,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> tuple[dict, Path]:
    """Resolve an experiment onto a server and create its immutable run record."""
    experiment = ExperimentConfig.load(config_path)
    _validate_current_environment(experiment)
    catalog = ServerCatalog.load(catalog_path)
    selected_server = server_name or experiment.data["simulator"].get("server")
    if not selected_server:
        raise ValueError("--server is required when the experiment does not select a server")
    run_dir = create_run(
        experiment,
        catalog,
        selected_server,
        runs_root,
        run_name=run_name,
        command=command,
        runtime_overrides=runtime_overrides,
    )
    return json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8")), run_dir


__all__ = ["DEFAULT_SERVER_CATALOG", "create_configured_run"]
