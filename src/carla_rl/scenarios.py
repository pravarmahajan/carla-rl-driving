"""Versioned, deterministic CARLA scenario specifications.

This module models placement and lifecycle ownership only.  NPC controllers,
Traffic Manager policies, reward changes, and perception remain separate work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class ScenarioConfigurationError(ValueError):
    """Raised when a scenario cannot be deterministically reconstructed."""


@dataclass(frozen=True)
class TransformSpec:
    x: float
    y: float
    z: float
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, field_name: str) -> "TransformSpec":
        if not isinstance(value, Mapping):
            raise ScenarioConfigurationError(f"{field_name} must be an object")
        required = ("x", "y", "z")
        missing = [key for key in required if key not in value]
        if missing:
            raise ScenarioConfigurationError(f"{field_name} is missing: {', '.join(missing)}")
        allowed = set(required) | {"pitch", "yaw", "roll"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ScenarioConfigurationError(f"{field_name} has unsupported fields: {', '.join(unknown)}")
        try:
            return cls(**{key: float(value.get(key, 0.0)) for key in allowed})
        except (TypeError, ValueError) as exc:
            raise ScenarioConfigurationError(f"{field_name} coordinates must be numeric") from exc

    def to_carla_transform(self, carla_api: Any) -> Any:
        return carla_api.Transform(
            carla_api.Location(x=self.x, y=self.y, z=self.z),
            carla_api.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll),
        )

    def to_carla_location(self, carla_api: Any) -> Any:
        return carla_api.Location(x=self.x, y=self.y, z=self.z)


@dataclass(frozen=True)
class ActorSpec:
    name: str
    blueprint: str
    transform: TransformSpec
    attributes: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int) -> "ActorSpec":
        if not isinstance(value, Mapping):
            raise ScenarioConfigurationError(f"actors[{index}] must be an object")
        required = {"name", "blueprint", "transform"}
        missing = sorted(required - value.keys())
        if missing:
            raise ScenarioConfigurationError(f"actors[{index}] is missing: {', '.join(missing)}")
        unknown = sorted(set(value) - (required | {"attributes"}))
        if unknown:
            raise ScenarioConfigurationError(f"actors[{index}] has unsupported fields: {', '.join(unknown)}")
        name, blueprint = value["name"], value["blueprint"]
        if not isinstance(name, str) or not name:
            raise ScenarioConfigurationError(f"actors[{index}].name must be a non-empty string")
        if not isinstance(blueprint, str) or not blueprint:
            raise ScenarioConfigurationError(f"actors[{index}].blueprint must be a non-empty string")
        attributes = value.get("attributes", {})
        if not isinstance(attributes, Mapping) or not all(
            isinstance(key, str) and isinstance(item, (str, int, float, bool))
            for key, item in attributes.items()
        ):
            raise ScenarioConfigurationError(f"actors[{index}].attributes must map strings to scalar values")
        return cls(
            name=name,
            blueprint=blueprint,
            transform=TransformSpec.from_mapping(value["transform"], field_name=f"actors[{index}].transform"),
            attributes={key: str(item) for key, item in attributes.items()},
        )


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    ego_start: TransformSpec | None = None
    goal: TransformSpec | None = None
    actors: tuple[ActorSpec, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScenarioSpec":
        if not isinstance(value, Mapping):
            raise ScenarioConfigurationError("scenario must be an object")
        missing = {"id"} - value.keys()
        if missing:
            raise ScenarioConfigurationError(f"scenario is missing: {', '.join(sorted(missing))}")
        unknown = sorted(set(value) - {"id", "ego_start", "goal", "actors"})
        if unknown:
            raise ScenarioConfigurationError(f"scenario has unsupported fields: {', '.join(unknown)}")
        scenario_id = value["id"]
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ScenarioConfigurationError("scenario.id must be a non-empty string")
        raw_actors = value.get("actors", [])
        if not isinstance(raw_actors, list):
            raise ScenarioConfigurationError("scenario.actors must be a list")
        actors = tuple(ActorSpec.from_mapping(item, index=index) for index, item in enumerate(raw_actors))
        if len({actor.name for actor in actors}) != len(actors):
            raise ScenarioConfigurationError("scenario actor names must be unique")
        return cls(
            scenario_id=scenario_id,
            ego_start=(TransformSpec.from_mapping(value["ego_start"], field_name="scenario.ego_start")
                       if "ego_start" in value else None),
            goal=(TransformSpec.from_mapping(value["goal"], field_name="scenario.goal")
                  if "goal" in value else None),
            actors=actors,
        )
