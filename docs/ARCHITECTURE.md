# CARLA RL architecture

This document describes the current execution architecture, not a target
autonomous-driving stack.

## System map

```text
scripts/carla
  ├─ validates the active Python environment
  ├─ leases a named CARLA server slot
  └─ launches train.py / eval.py / drive.py
       ├─ loads configs/experiments/<experiment>.json
       ├─ creates runs/<run-id>/ manifest + resolved config + git.patch
       └─ constructs CarlaGymEnv
            ├─ CARLA Python API client
            ├─ CARLA route planner from $CARLA_ROOT
            ├─ state-vector observation and vehicle-control action spaces
            └─ reward, termination, actor, and synchronous-world lifecycle

CARLA Docker containers
  sim-0 (2000)   sim-1 (2010)   sim-2 (2020)
```

## Source components

| Component | Responsibility |
| --- | --- |
| `carla_gym_env.py` | Current Gymnasium environment: CARLA connection, actor/sensor lifecycle, route generation, state observation, action application, reward, and termination. |
| `train.py` | Stable-Baselines3 PPO training, normalization, checkpoints, callback metrics. |
| `eval.py` | Deterministic policy evaluation and aggregate metrics. |
| `drive.py` | Interactive pygame/camera visualization of a policy. |
| `src/carla_rl/config.py` | Versioned experiment and server-catalog validation. |
| `src/carla_rl/server_slots.py` | Process-local exclusive lease for one named simulator slot. |
| `src/carla_rl/run_manifest.py` | Immutable run provenance: resolved config, dependency versions, Git revision/status/patch, and CARLA source provenance. |
| `src/carla_rl/launch.py` | Shared config-backed launch validation and run creation. |
| `infra/carla-compose.yaml` | Three independent CARLA 0.9.16 Docker services and their host port mappings. |

## Execution lifecycle

1. Start the required simulator slot, for example `scripts/carla cluster up sim-0`.
2. `scripts/carla train` or `evaluate` takes an exclusive file lock for that
   slot. A second repository client cannot accidentally use the same slot.
3. The launcher loads the versioned experiment config and resolves the selected
   slot to host/ports.
4. Before connecting to CARLA, it creates `runs/<run-id>/` containing
   `manifest.json`, `resolved_config.json`, and `git.patch`.
5. `CarlaGymEnv` connects to CARLA and sets synchronous mode. The policy then
   acts through the environment.
6. Training saves model and VecNormalize state inside its run directory;
   evaluation saves `metrics.json`; driving saves `drive_results.json`.

`scripts/carla doctor` reads Docker container state and the wrapper's lease
records to show which slots are up and which are currently owned by a live
repository client. A manually started CARLA client is outside this visibility.

## Server-slot policy

The three servers use the same CARLA image and settings. Their roles are
conventions:

- `sim-0`: default training slot.
- `sim-1`: spare training or evaluation slot.
- `sim-2`: default dedicated evaluation/driving slot.

Keeping evaluation separate prevents training resets and world ticks from
interfering with evaluation. If no training client is using `sim-1`, it is
valid to evaluate there with `scripts/carla evaluate --slot sim-1 ...`.

## Reproducibility boundary

Configuration controls settings the current code implements: server selection,
CARLA timestep, action repeat, steering filter, episode limit, PPO parameters,
and seed. Git controls code changes. A run can use uncommitted code; its
manifest records the base commit and patch. Prefer a clean commit for a result
you want to promote as a baseline.

The current environment intentionally supports only the recorded nine-element
`state-v1` observation layout and three-element vehicle-control action layout.
If a config asks for a different feature list—for example `speed_limit_kmh`—the
launcher fails rather than silently run a mismatched policy. The planned
observation-registry refactor will turn such features into explicit modules.

## Deterministic scenario boundary

`environment.scenario` is an optional, versioned initial-world-state contract.
It can select a named scenario ID, fixed ego start and goal, and named non-ego
actors with explicit CARLA blueprint, transform, and blueprint attributes. The
environment owns every listed actor and destroys it on reset or close. Scenario
ID and actor count are emitted in Gymnasium `reset()`/`step()` info.

This boundary intentionally does **not** implement actor motion, Traffic
Manager/autopilot, traffic-light scheduling, pedestrians' AI controllers,
reward changes, or perception inputs. Those behaviors must be added as a
separate scenario/controller contract so an experiment can isolate them.

```json
"scenario": {
  "id": "example-static-lead-v0",
  "ego_start": {"x": 0.0, "y": 0.0, "z": 0.5, "yaw": 0.0},
  "goal": {"x": 120.0, "y": 0.0, "z": 0.5},
  "actors": [{
    "name": "lead_vehicle",
    "blueprint": "vehicle.tesla.model3",
    "transform": {"x": 20.0, "y": 0.0, "z": 0.5, "yaw": 0.0},
    "attributes": {"role_name": "scenario_lead"}
  }]
}
```

The coordinates above illustrate the schema only; use transforms verified on
the selected CARLA map. The launchers pass this config to `CarlaGymEnv`, and
the resolved configuration is recorded in the run manifest.

## External CARLA source dependency

The `carla` wheel supplies the core Python API, while
`agents.navigation.global_route_planner` is imported from
`$CARLA_ROOT/PythonAPI/carla/agents` (default `$HOME/git/carla`). Therefore the
CARLA source checkout is executable dependency code. Each manifest records its
Git commit and whether its `agents` subtree was dirty.

## Artifact policy

`runs/` contains new run artifacts and is Git-ignored. Historical artifacts
from before this platform live in `artifacts/legacy/2026-08-30/`; they are
recoverable but also Git-ignored. Baseline metadata in `baselines/` points to
the relevant archived artifacts and stores checksums.
