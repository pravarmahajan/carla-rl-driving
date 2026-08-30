# CARLA RL runbook

This is the operational reference for running the current repository. It does
not replace `THINKING.md` for a scientific experiment.

## Environment

Use the already-created Conda environment:

```bash
conda activate carla_env
scripts/carla doctor
```

`environment.yml` records the dependency set for a new machine. Do not rerun
it for ordinary work in this checkout. The optional `.envrc` exports repository
paths if `direnv` is installed and allowed; it intentionally does not activate
Conda.

## CARLA server cluster

```bash
scripts/carla cluster up
scripts/carla cluster status
scripts/carla cluster logs sim-0
scripts/carla cluster down
```

`cluster up` starts three independent project-owned CARLA Docker containers:

| Slot | Host ports | Intended role |
| --- | --- | --- |
| `sim-0` | 2000-2002 | training |
| `sim-1` | 2010-2012 | future parallel training |
| `sim-2` | 2020-2022 | evaluation and visual driving |

It only starts simulators; it does not start a policy or training process.
`cluster down` stops and removes only this Compose project's CARLA containers
and network. It does not remove Docker images, checkpoints, runs, or unrelated
containers.

## Commands

```bash
scripts/carla train
scripts/carla evaluate --model-path runs/<training-run-id>/model
scripts/carla drive --model-path runs/<training-run-id>/model
./start_tensorboard.sh
```

The wrapper leases `sim-0` for training and `sim-2` for evaluation or driving.
Every config-backed command writes a new `runs/<run-id>/` directory before
connecting to CARLA. That directory contains the resolved config, Git state,
dependency versions, CARLA route-planner provenance, and command metadata.

Set `CARLA_RL_CONFIG=/absolute/path/to/config.json` to select a different
checked-in experiment configuration.

## Archived baseline

The current candidate frozen baseline is
`baselines/state_v1_round16b.json`. Its checkpoint and matching VecNormalize
state are stored at:

```text
artifacts/legacy/2026-08-30/checkpoints/ppo_carla_model.zip
artifacts/legacy/2026-08-30/checkpoints/ppo_carla_model_vecnormalize.pkl
```

Use that pair together:

```bash
scripts/carla evaluate \
  --model-path artifacts/legacy/2026-08-30/checkpoints/ppo_carla_model
```

This is a compatibility/evaluation run, not an exact historical episode replay.
The old project did not record a fixed route, evaluation seed, exact server map
state, or a checkpoint-to-success mapping. The last explicitly documented
standalone success result is Round 13 (6/10 random-route evaluation episodes),
whereas the archived Round 16b checkpoint remains a candidate pending a new
fixed-route benchmark.

## Before a real experiment

Write the observation, hypothesis, predicted outcome, and a single causal
question in `THINKING.md`. Then use a new versioned config and preserve the
resulting run directory. Do not overwrite a checkpoint in place.
