# CARLA RL Driving Agent — Progress Log

PPO (Stable-Baselines3) agent learning to drive a route in CARLA, using
`CarlaGymEnv` (`carla_gym_env.py`) as the Gym environment. Trained via
`train.py`, watched via `drive.py` (pygame camera view), scored via `eval.py`.

Requires `CARLA_ROOT` env var pointing at a CARLA repo checkout if it's not at
the default `~/git/carla` (used to locate `agents.navigation.global_route_planner`).

## Timeline

- **Round 1**: initial 4-observation-dim environment (speed, distance/angle to
  next waypoint, distance to goal). Baseline behavior only.
- **Round 2 (collapsed)**: ran on stale buggy stall-detection code still loaded
  in memory; reward collapsed to ~-9.98 with 1-4 step episodes for ~2700
  episodes — wasted compute, no checkpoint saved mid-run. Root-caused after
  the fact; not a reward-design issue, an execution bug. Restored from
  `ppo_carla_model_backup_before_round2.zip` afterward.
- **Round 3**: fresh restart post-fix, 100k timesteps, completed clean.
- **Round 4**: added 6-dim observation space (+ lane_offset, heading_error),
  lane-invasion sensor with -2.0 penalty on solid-marking crossings, off-road
  detection via `get_waypoint(project_to_road=False) is None`, wrong-way
  detection via OpenDrive lane_id sign flip, harsher terminal penalties
  (crash -100, off-road/wrong-way -75, stall -10, timeout -50), periodic
  checkpoint saving every 10k steps (`PeriodicSaveCallback` in train.py) so
  `drive.py` always has the latest weights and crashes don't lose progress.
  Requested 150k timesteps; crashed at ~129,590 due to a CARLA server
  timeout (infra hiccup, not a code bug) but the periodic-save callback
  preserved progress. Reward/length trended upward across the run
  (159 → 257 → 296 avg reward across episode buckets) — no plateau evidence.

- **Round 5 (complete)**: implemented all of "Reward/feature fixes" (#1
  below): potential-based distance-to-goal shaping (`0.5 * (prev_dist - dist)`,
  replacing the flat +10 waypoint bonus with a small +2 milestone bonus),
  continuous per-tick lane_offset/heading_error penalty, previous
  steer/throttle action appended to the observation (now 8-dim), multi-
  waypoint lookahead (circular mean of heading error over next 5 waypoints
  instead of just the next one), and `termination_reason` in `step()`'s info
  dict + `EpisodeLoggerCallback` (crash/off_road/wrong_way/stall/timeout/success).
  Observation space shape changed (6→8 dims), so `ppo_carla_model.zip` from
  round 4 is incompatible — started fresh with `train.py --fresh` (log:
  `train_round5.log`). Ran the full 50k timesteps cleanly (341 episodes),
  model saved. `termination_reason` breakdown worked end-to-end (off_road and
  wrong_way dominate; crash and stall much rarer) — first run with an exact
  cause-of-death signal instead of inferring it from reward magnitude.
  Note: the local TensorBoard instance had been left pointed at the old
  pre-migration path (`.../PythonAPI/personal/logs/`) and needed restarting
  against this repo's `./logs/` to show round 5's `PPO_1` run.

## Known issues identified (as of round 4)

1. **Reward imbalance at longer routes**: flat +10 waypoint bonus means a
   150m route can accumulate ~750 in waypoint bonuses alone, which can
   outweigh even the -100/-75 terminal penalties. A policy that drives 90%
   of a route then crashes can still net positive reward.
2. **Single-waypoint observation noise**: `_get_observation()` only looks at
   the *next* 2m waypoint, so marker-placement jitter on straight roads can
   look like a steering-relevant angle change when it isn't — contributes to
   the "unnecessary turning on straights" behavior.
3. **No previous-action feature**: the agent has no signal about its own
   recent control history, so it can't detect/dampen its own oscillation
   (relevant to the left-turn bias / circling behavior observed in one run).
4. **No exact termination-reason logging**: `episode_log.txt` records
   reward/length but not *why* an episode ended (crash vs. off-road vs.
   wrong-way vs. stall vs. timeout vs. success). Currently only inferable
   heuristically from reward magnitude — not exact.

## Planned next steps (prioritized)

1. **Reward/feature fixes (do first, no algorithm risk)**:
   - Replace flat waypoint bonus with potential-based shaping:
     `reward += k * (prev_distance_to_goal - distance_to_goal)`, which
     telescopes to a route-length-independent total, keeping terminal
     penalties meaningful regardless of route length.
   - Add a small continuous per-tick penalty on `lane_offset`/`heading_error`
     (already computed, currently unused in the reward function).
   - Add previous steering/throttle action to the observation vector.
   - Average heading/curvature over the next 3-5 waypoints instead of just
     the next one, to reduce noise on straight roads.
   - Add explicit `termination_reason` to `step()`'s info dict + log it in
     `EpisodeLoggerCallback` for exact crash/off-road/wrong-way/stall/timeout/
     success breakdowns in future runs.
2. **Try SAC in parallel** (not a replacement for PPO, an experiment):
   CARLA's synchronous 20Hz tick makes each environment step wall-clock
   expensive; PPO is on-policy and discards each 2048-step rollout after
   `n_epochs=10` gradient passes, while SAC is off-policy (replay buffer,
   reuses old transitions) and auto-tunes its entropy temperature — a
   structural fix for the left-turn-bias/mode-collapse concern instead of
   the current fixed `ent_coef` band-aid.
3. **Architecture/optimizer changes**: currently deprioritized — no evidence
   of a capacity/optimization bottleneck (reward was still climbing steadily
   when round 4 crashed).

## Repo layout notes

- Model checkpoints (`*.zip`), tensorboard `logs/`, `episode_log.txt`, and
  run logs (`*.log`, `*.pid`) are gitignored — they're regenerated by
  training runs, not source. They still live in this directory locally.
- `ppo_carla_model.zip` is the canonical checkpoint `train.py`/`drive.py`/
  `eval.py` load from by default.
