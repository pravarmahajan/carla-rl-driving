"""Explicit, independently testable building blocks for ``CarlaGymEnv``.

These classes intentionally do not decide an experiment's feature layout.  They
preserve the established state-v1 behaviour while separating CARLA ownership,
observation construction, reward calculation, and episode completion policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np


class CarlaSession:
    """Own one CARLA client connection and the server-wide world settings.

    CARLA settings are server-wide, so this object is deliberately the only
    component allowed to apply or restore them.  The environment keeps public
    aliases to its world/map for compatibility with ``drive.py``.
    """

    def __init__(
        self,
        carla_api: Any,
        route_planner_cls: Any,
        *,
        host: str,
        port: int,
        town: Optional[str],
        fixed_delta_seconds: float,
        no_rendering: bool,
    ) -> None:
        self._carla = carla_api
        self.client = carla_api.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town) if town else self.client.get_world()
        self.map = self.world.get_map()
        self.town = self.map.name
        self.blueprint_library = self.world.get_blueprint_library()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        settings.no_rendering_mode = no_rendering
        self.world.apply_settings(settings)

        self.route_planner = route_planner_cls(self.map, sampling_resolution=2.0)
        self.static_vehicle_bboxes = list(
            self.world.get_level_bbs(carla_api.CityObjectLabel.Car)
        )

    def close(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)


@dataclass(frozen=True)
class ObservationInputs:
    """Numeric state consumed by the frozen state-v1 observation contract."""

    speed_kmh: float
    distance_to_waypoint_m: float
    route_heading_error_rad: float
    distance_to_goal_m: float
    lane_offset_m: float
    lane_heading_error_rad: float
    previous_steer: float
    previous_throttle: float
    obstacle_distance_m: float

    def as_state_v1(self) -> np.ndarray:
        return np.array(
            [
                self.speed_kmh,
                self.distance_to_waypoint_m,
                self.route_heading_error_rad,
                self.distance_to_goal_m,
                self.lane_offset_m,
                self.lane_heading_error_rad,
                self.previous_steer,
                self.previous_throttle,
                self.obstacle_distance_m,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class RewardInputs:
    """Inputs to round-12-v1 reward, with no CARLA actor dependency."""

    speed_kmh: float
    throttle: float
    brake: float
    previous_distance_to_goal_m: float
    distance_to_goal_m: float
    distance_to_waypoint_m: float
    route_heading_error_rad: float
    lane_offset_m: float
    lane_heading_error_rad: float
    obstacle_distance_m: float
    lane_invaded: bool
    reached_waypoint: bool


class Round12Reward:
    """The frozen baseline's reward function, isolated from simulator I/O."""

    obstacle_lookahead_m = 30.0

    def evaluate(self, inputs: RewardInputs) -> float:
        reward = -0.01
        if inputs.speed_kmh < 1.0 and inputs.throttle < 0.1:
            reward -= 0.3
        reward += inputs.previous_distance_to_goal_m - inputs.distance_to_goal_m
        if inputs.reached_waypoint:
            reward += 2.0
        if inputs.speed_kmh > 1.0:
            reward += max(0.0, 0.5 - abs(inputs.route_heading_error_rad) / np.pi) * 0.1
        reward -= 0.15 * abs(inputs.lane_offset_m)
        reward -= 0.1 * abs(inputs.lane_heading_error_rad)
        if inputs.lane_invaded:
            reward -= 2.0
        if inputs.obstacle_distance_m < self.obstacle_lookahead_m and inputs.speed_kmh > 2.0:
            danger = (self.obstacle_lookahead_m - inputs.obstacle_distance_m) / self.obstacle_lookahead_m
            reward -= danger * inputs.speed_kmh * 0.02
            reward += danger * inputs.brake * 0.3
        return reward


@dataclass(frozen=True)
class TerminationInputs:
    crashed: bool
    off_road: bool
    wrong_way: bool
    stall_counter: int
    reached_goal: bool
    episode_steps: int
    max_episode_steps: int


@dataclass(frozen=True)
class EpisodeOutcome:
    terminated: bool
    truncated: bool
    reason: Optional[str]
    reward_adjustment: float


class Round12Termination:
    """The frozen baseline's terminal-state precedence and penalties."""

    def evaluate(self, inputs: TerminationInputs) -> EpisodeOutcome:
        outcome: EpisodeOutcome
        if inputs.crashed:
            outcome = EpisodeOutcome(True, False, "crash", -30.0)
        elif inputs.off_road:
            outcome = EpisodeOutcome(True, False, "off_road", -30.0)
        elif inputs.wrong_way:
            outcome = EpisodeOutcome(True, False, "wrong_way", -30.0)
        elif inputs.stall_counter >= 20:
            outcome = EpisodeOutcome(True, False, "stall", -30.0)
        elif inputs.reached_goal:
            outcome = EpisodeOutcome(True, False, "success", 500.0)
        else:
            outcome = EpisodeOutcome(False, False, None, 0.0)
        # This looks unusual, but it characterizes the legacy environment:
        # its independent time-limit check runs after all terminal checks.
        # Preserve that exact reason/penalty precedence before changing it in
        # a deliberately designed future experiment.
        if inputs.episode_steps >= inputs.max_episode_steps:
            return EpisodeOutcome(outcome.terminated, True, "timeout", outcome.reward_adjustment - 20.0)
        return outcome


def heading_error_rad(vehicle_yaw_deg: float, lane_yaw_deg: float) -> float:
    """Return the state-v1 signed heading error, normalized to [-pi, pi]."""
    heading_diff = ((vehicle_yaw_deg - lane_yaw_deg + 180.0) % 360.0) - 180.0
    return heading_diff * math.pi / 180.0
