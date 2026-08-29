# CARLA RL Driving

This repository trains and evaluates a Stable-Baselines3 PPO driving policy
against CARLA. The current reproducible baseline uses a nine-value state vector,
a three-value continuous control action, and CARLA 0.9.16.

Create the recorded Python environment with `conda env create -f environment.yml`.
The environment also imports CARLA navigation agents from `CARLA_ROOT` (default
`~/git/carla`); run manifests record that checkout's Git revision because it is
executable source, not merely data.

## Repository roles

- `THINKING.md`: researcher-owned observations, hypotheses, predictions, and
  proposed experiments. Do not use it as a task list.
- `PROGRESS.md`: append-only factual experiment history.
- `docs/LINEAR_BACKLOG.md`: execution backlog ready to copy into Linear.
- `configs/`: versioned simulator and experiment configuration.
- `baselines/`: metadata for important code/checkpoint pairs. Model binaries
  remain outside Git.

## Known-good baseline

The pre-speed-limit baseline is recorded in
`baselines/state_v1_round16b.json`. Its source revision is `5a3a17b`; the
checkpoint has nine observations, three actions, and 303,104 PPO timesteps.

## CARLA server slots

The repository defines three independent servers:

| Slot | RPC ports | Traffic Manager port | Default role |
| --- | --- | --- | --- |
| `sim-0` | 2000-2002 | 8000 | training |
| `sim-1` | 2010-2012 | 8010 | training |
| `sim-2` | 2020-2022 | 8020 | evaluation/driving |

Start or inspect only this project's CARLA containers:

```bash
scripts/carla-cluster up
scripts/carla-cluster status
scripts/carla-cluster logs sim-0
scripts/carla-cluster down
```

The launcher is intentionally scoped to `infra/carla-compose.yaml`; it never
stops unrelated Docker containers.

Run a client while holding an exclusive simulator lease:

```bash
PYTHONPATH=src python -m carla_rl.server_slots --slot sim-0 -- \
  python train.py --port 2000

PYTHONPATH=src python -m carla_rl.server_slots --slot sim-2 -- \
  python drive.py --port 2020
```

Use `--slot auto` to take the first available slot. The wrapper exports
`CARLA_HOST`, `CARLA_PORT`, `CARLA_TM_PORT`, and `CARLA_SLOT` to the child
process. Explicit CLI ports remain supported until the training and driving
entry points are migrated to the shared configuration module.

## Run manifests

Create an immutable run directory before launching an experiment:

```bash
PYTHONPATH=src python -m carla_rl.run_manifest \
  --config configs/experiments/baseline_state_v1.json \
  --server sim-0 \
  --run-name baseline-smoke
```

The command writes the resolved configuration, Git revision/dirty state,
dependency versions, server selection, and a source diff into `runs/<run-id>/`.
It refuses to overwrite an existing run.

## Tests

The initial infrastructure tests use only the Python standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

CARLA and GPU integration tests will be added separately and marked so the unit
suite remains runnable without a simulator.
