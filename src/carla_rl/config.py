"""Load and validate versioned CARLA server and experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when a configuration is incomplete or internally inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Top-level configuration must be an object: {path}")
    return data


def canonical_fingerprint(data: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible configuration."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ServerConfig:
    name: str
    host: str
    rpc_port: int
    streaming_port: int
    secondary_port: int
    traffic_manager_port: int
    default_role: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ServerConfig":
        required = {
            "name",
            "host",
            "rpc_port",
            "streaming_port",
            "secondary_port",
            "traffic_manager_port",
            "default_role",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ConfigurationError(f"Server is missing fields: {', '.join(missing)}")
        server = cls(**{key: value[key] for key in required})
        ports = [server.rpc_port, server.streaming_port, server.secondary_port, server.traffic_manager_port]
        if any(not isinstance(port, int) or not 1 <= port <= 65535 for port in ports):
            raise ConfigurationError(f"Server {server.name!r} contains an invalid port")
        if len(set(ports)) != len(ports):
            raise ConfigurationError(f"Server {server.name!r} reuses a port")
        return server

    def as_environment(self) -> dict[str, str]:
        return {
            "CARLA_SLOT": self.name,
            "CARLA_HOST": self.host,
            "CARLA_PORT": str(self.rpc_port),
            "CARLA_TM_PORT": str(self.traffic_manager_port),
        }


@dataclass(frozen=True)
class ServerCatalog:
    schema_version: int
    servers: tuple[ServerConfig, ...]
    source: Path
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> "ServerCatalog":
        source = Path(path).resolve()
        data = _read_json(source)
        if data.get("schema_version") != 1:
            raise ConfigurationError("Unsupported server catalog schema_version")
        raw_servers = data.get("servers")
        if not isinstance(raw_servers, list) or not raw_servers:
            raise ConfigurationError("Server catalog must contain a non-empty servers list")
        servers = tuple(ServerConfig.from_mapping(item) for item in raw_servers)
        names = [server.name for server in servers]
        if len(names) != len(set(names)):
            raise ConfigurationError("Server names must be unique")
        allocated_ports = [
            port
            for server in servers
            for port in (
                server.rpc_port,
                server.streaming_port,
                server.secondary_port,
                server.traffic_manager_port,
            )
        ]
        if len(allocated_ports) != len(set(allocated_ports)):
            raise ConfigurationError("Ports must be unique across all server slots")
        return cls(1, servers, source, canonical_fingerprint(data))

    def get(self, name: str) -> ServerConfig:
        for server in self.servers:
            if server.name == name:
                return server
        choices = ", ".join(server.name for server in self.servers)
        raise ConfigurationError(f"Unknown server {name!r}; choose one of: {choices}")


@dataclass(frozen=True)
class ExperimentConfig:
    data: dict[str, Any]
    source: Path
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path).resolve()
        data = _read_json(source)
        if data.get("schema_version") != 1:
            raise ConfigurationError("Unsupported experiment schema_version")
        for field in ("name", "research", "simulator", "environment", "algorithm", "evaluation"):
            if field not in data:
                raise ConfigurationError(f"Experiment is missing field: {field}")

        environment = data["environment"]
        observations = environment.get("observation_features")
        actions = environment.get("action_features")
        if not isinstance(observations, list) or not observations:
            raise ConfigurationError("observation_features must be a non-empty list")
        if len(observations) != len(set(observations)):
            raise ConfigurationError("observation_features contains duplicates")
        if not isinstance(actions, list) or not actions:
            raise ConfigurationError("action_features must be a non-empty list")
        if len(actions) != len(set(actions)):
            raise ConfigurationError("action_features contains duplicates")
        return cls(data, source, canonical_fingerprint(data))

