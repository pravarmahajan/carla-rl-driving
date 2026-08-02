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

- **Round 6 (in progress)**: continued training from round 5's checkpoint for
  200k more timesteps (`train.py --total-timesteps 200000`, no `--fresh` —
  same 8-dim obs space, so the checkpoint loads directly), to see if reward
  keeps improving with more exposure now that the network isn't starting from
  scratch. Reason: round 5 alone (50k steps, 341 episodes) still had 90% of
  episodes ending in off_road/wrong_way and only 1 success, but that's
  expected to be undertrained rather than a reward-design flaw given the obs
  space reset. Log: `train_round6.log`.
- **Dashboard review during round 6 / planned round 7 changes**: watching
  TensorBoard mid-run raised two real gaps:
  - `train/value_loss` isn't decreasing — likely because observations aren't
    normalized (obs dims span wildly different scales: speed 0-50,
    distance 0-500, angles -π..π, actions -1..1), which is a classic cause of
    a value net struggling to fit. Fix: wrap the env in SB3's `VecNormalize`.
  - No held-out/deterministic evaluation signal distinct from the noisy
    per-episode training rollout stats. Fix: a periodic deterministic-eval
    callback, logged to a separate `eval/` TensorBoard tab.
  Both implemented in `train.py` for round 7 onward (not applied retroactively
  to the in-progress round 6 run):
  - `VecNormalize` (`norm_obs=True, norm_reward=True, clip_obs=10.0`) wrapping
    a `DummyVecEnv`-wrapped `CarlaGymEnv`. Stats saved/loaded alongside the
    model as `<model_path>_vecnormalize.pkl` (gitignored, regenerated).
    `eval.py`/`drive.py` updated to load these stats and normalize
    observations before `model.predict()` — required for correctness, since a
    policy trained on normalized inputs behaves wrongly if fed raw ones at
    inference time.
  - `PeriodicEvalCallback`: runs 3 deterministic episodes every 5 rollouts
    (~10k timesteps) on the *same* training env/vehicle (not a second one —
    CARLA's synchronous `world.tick()` advances every actor at once
    regardless of which client calls it, so a second vehicle ticking the
    world mid-rollout would drag the idle training vehicle along with it).
    Logs `eval/mean_reward`, `eval/mean_length`, `eval/success_rate` to a
    separate TensorBoard tab from the noisy per-episode `episode/*` stats.

- **Round 7 (complete, big improvement)**: fresh run (`--fresh`, new
  `VecNormalize` wrapping changes the input distribution enough that resuming
  round 6's checkpoint would've been effectively a reset anyway) with the
  `VecNormalize` + `PeriodicEvalCallback` changes above. Log: `train_round7.log`.
  Hit a real bug on first launch: manually building the `VecEnv` ourselves
  (needed to insert `VecNormalize`) meant SB3 no longer auto-wrapped the env
  in `Monitor` (that auto-wrap only happens when you hand PPO a raw `gym.Env`,
  not an already-vectorized one) — so no `info["episode"]` key ever got set,
  silently breaking `EpisodeLoggerCallback` *and* SB3's own
  `rollout/ep_rew_mean` stats (confirmed via TensorBoard's event tags: only
  `train/*` and our `eval/*` existed, no `rollout/*`). Fixed by explicitly
  wrapping `Monitor(CarlaGymEnv())` before `DummyVecEnv`, then restarted.
  Early results after the fix (~20k timesteps in) are the best yet:
  `train/value_loss` actually low with `explained_variance≈0.67` (vs. never
  decreasing before `VecNormalize`), `eval/mean_length` far longer than any
  prior round, and qualitatively (per user observation driving it) it now
  **stays in its lane and follows curves properly** instead of drifting
  straight through them — the `VecNormalize` + reward-shaping combination
  seems to have fixed the "ignores curves" problem from round 5/6. Not
  reaching the destination yet, and per user observation it also doesn't
  execute turns at intersections (as opposed to following a gradual curve) --
  suspect #2 below (limited waypoint lookahead) or that sharper turns need
  more advance warning than gradual curves do. Full run: 150k timesteps,
  ended with episode lengths in the 600-960+ step range (vs. ~100-200 in
  rounds 5/6) and rewards >1500 on some episodes — model + vecnormalize stats
  saved.
- **Post-round-7 tooling additions**:
  - `PeriodicEvalCallback` now also logs `eval/value_loss`: for each
    deterministic eval step, compares the critic's predicted value against
    the empirical discounted return actually realized for the rest of that
    episode (computed from the un-normalized rewards, treating truncation
    like termination as an approximation). This is a value-loss signal
    computed on held-out, deterministic data — distinct from
    `train/value_loss`, which only ever sees noisy on-policy GAE targets from
    the training rollout itself.
  - `drive.py` now renders a top-down `MiniMap` in the corner of the window
    (route line, start, goal, live vehicle position/heading) alongside the
    camera feed, since the camera alone gives no sense of progress toward
    the destination or overall route shape — added specifically to help
    diagnose the "doesn't reach the destination / doesn't turn at
    intersections" behavior from round 7.

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
