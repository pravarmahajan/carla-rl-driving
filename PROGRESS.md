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
  reaching the destination yet. Correction to an earlier note in this log:
  the car *does* attempt turns at intersections (that first impression was
  wrong, likely formed while drive.py's fixed-start spawn bug was still
  making attempts land in confusing/inconsistent spots) -- the real pattern,
  confirmed after fixing drive.py and testing properly, is that episodes
  often end mid-turn or shortly after one (terminating well short of the
  1500-step cap, via an actual off_road/wrong_way/crash condition, not a
  timeout) rather than failing to turn at all. Full run: 150k timesteps,
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
  - Fixed `drive.py`'s fixed-start spawn: a resting vehicle transform sits a
    hair below true ground level (suspension compression), so re-spawning a
    fresh vehicle at that exact z collided with the static road mesh itself
    and silently fell back to a random spawn point every time -- fixed by
    lifting the spawn z slightly and letting the existing settle-tick loop
    drop it back down. Also switched drive.py to a fresh random route per
    attempt instead of repeating one fixed route 3x, since the policy +
    environment are deterministic and repeating an identical scenario just
    reproduced the same trajectory every time. `drive.py` now also prints the
    exact `termination_reason` per attempt instead of a coarser
    reached-goal/crashed guess.
  - **Found and fixed a wrong_way false-positive bug** using the corrected
    per-attempt diagnostics: `termination_reason=wrong_way` was firing
    immediately after legitimate turns at intersections, which is what had
    looked like "doesn't turn" / "terminates as soon as it turns" in earlier
    testing. Root cause: wrong-way detection compared the current lane's
    OpenDrive `lane_id` sign against the sign recorded once at `reset()` --
    but that sign convention only holds within a single road segment, not
    across the whole route, so turning onto a different road at an
    intersection could flip it and falsely flag a correct turn as wrong-way.
    Replaced with a per-step check comparing the vehicle's heading against
    its *current* lane's own local direction (`heading_error` from
    `_get_observation()`); >90 degrees off means driving against that lane's
    traffic flow, with no cross-road dependency. This is an environment/eval
    fix, not a policy change -- round 7's checkpoint should be re-evaluated
    with `drive.py` under the corrected logic before deciding whether
    retraining is actually needed.
  - Still firing on `wrong_way` after that fix. Added debug logging (road_id,
    lane_id, `is_junction`, lane heading, vehicle heading, heading_error) on
    every `wrong_way` trigger, and confirmed via user testing that it's
    consistently `is_junction=True`: a junction's connector lanes curve
    rapidly, so the projected lane's local heading swings through the turn,
    making a normal, correct turn transiently look >90 degrees "wrong" by
    the same per-lane heading check. Fixed by skipping the wrong-way check
    entirely while inside a junction -- `off_road` detection and the
    lane-invasion sensor still catch bad driving once the vehicle exits back
    onto a normal road segment. Also an environment/eval fix, not a policy
    change.

- **Round 8 (complete)**: continued training from round 7's checkpoint
  (`train.py --total-timesteps 150000`, no `--fresh` -- obs space and
  `VecNormalize` stats are unchanged, only the environment's wrong_way logic
  changed). Rationale: round 7's checkpoint was trained entirely under the
  old buggy wrong_way logic, which likely punished correct intersection turns
  repeatedly during training (not just eval) -- resuming lets the policy
  correct that learned avoidance habit under the fixed reward signal while
  keeping the lane-keeping/curve-following behavior it already learned,
  rather than re-learning everything from scratch. Log: `train_round8.log`.
  Early signs were good (first 4 episodes after resuming showed off_road/crash
  terminations only, no wrong_way at all, including one episode reaching
  1465/1500 steps before a crash), but the full run surfaced **reward
  hacking**: `episode_log.txt` shows crash/wrong_way episodes netting
  +4000-4700 total reward (e.g. `reward=4732.67, reason=crash`), because a
  flat per-tick `speed * 0.1` reward let the policy earn large reward just by
  driving fast and surviving long, dwarfing the -75/-100 terminal penalties.
  Net behavior: car survives indefinitely without crashing but doesn't reach
  the destination -- optimizing for "stay alive and collect speed reward,"
  not "finish the route." Also: `eval/value_loss` read ~1e5 vs.
  `train/value_loss` ~0.1 -- root-caused as a units bug in
  `PeriodicEvalCallback` (comparing the critic's normalized-scale
  predictions against a Monte-Carlo return built from raw, un-normalized
  rewards); fixed by running rewards through `venv.normalize_reward()`
  before accumulating the eval return.

- **Round 9 (complete)**: fresh run (`--fresh` -- obs/action space changed,
  so round 8's checkpoint doesn't load), implementing four fixes together:
  1. **Reward hacking fix**: removed the flat `speed * 0.1` per-tick reward
     entirely; replaced with potential-based distance-to-goal shaping
     (already present) as the only per-tick progress driver, so total reward
     is bounded by net distance closed rather than survival time.
  2. **Eval/train value_loss units fix** (above).
  3. **Steering-smoothness penalty**: `reward -= 0.2 * abs(steer_action -
     previous_steer)`, to address jittery left-right steering on otherwise
     straight roads.
  4. **Obstacle detection + braking**: added a `brake` action (3rd action
     dim), a 9th observation dim for distance to the nearest obstacle ahead
     (`_distance_to_obstacle_ahead()` -- checks both static level geometry
     via `get_level_bbs()`, since parked-car props baked into the map never
     show up in `world.get_actors()`, and any dynamic vehicle actors), and
     reward shaping that penalizes closing fast on a near obstacle while
     rewarding proportional braking.
  Terminal magnitudes also bumped up (crash -150, off_road/wrong_way -100,
  stall -20, timeout -75, success +20→+250) so the terminal outcome stays a
  dominant signal now that per-tick reward is small and bounded. Log:
  `train_round9.log`. Ran the full 150k timesteps, but surfaced a new failure
  mode: the policy **collapsed to standing still** (or barely creeping,
  steering back and forth in place) rather than driving --
  `episode_log.txt`'s back half is dominated by `reason=stall` and
  suspiciously small-magnitude `timeout` rewards (e.g. `-15.47`, `-7.47`).
  Root-caused to two things: (a) `ent_coef=0.0` (no exploration pressure) let
  the policy prematurely converge onto this degenerate low-risk mode once
  found, with nothing pushing it back out; (b) the stall check
  (`speed < 0.1 and throttle > 0.5`) was gameable -- a policy applying
  throttle in the 0.1-0.5 band while essentially stationary never tripped it
  at all, riding out the full 1500-step timeout instead, and even when stall
  *did* fire, its -20 penalty was far cheaper than risking the -100/-150
  terminal penalties of actually attempting to drive.

- **Round 10 (complete)**: resumed from round 9's checkpoint (no
  `--fresh` -- obs/action space unchanged from round 9), 150k timesteps,
  `--ent-coef 0.01`, targeting the round 9 stall-collapse directly:
  1. **Entropy bonus**: `--ent-coef 0.01` (was 0.0, the SB3 default) to keep
     exploration pressure alive and prevent premature convergence onto a
     degenerate low-risk policy.
  2. **Stall-check loophole closed**: dropped the `throttle > 0.5` condition
     -- now sustained `speed < 0.1` alone counts as stalled, regardless of
     throttle level, so there's no throttle value that lets the car sit
     still indefinitely undetected.
  3. **Unified bad-outcome penalties**: crash/off_road/wrong_way/stall all
     now -150 (was -150/-100/-100/-20); timeout stays -75; success bumped
     20/250 → +500. Previously stall (-20) was cheap relative to the other
     terminals, making "stand still and eat the small stall penalty" look
     safer than actually attempting to drive; equalizing removes that
     asymmetry.
  Chose to resume rather than restart from scratch since obs/action space is
  unchanged from round 9 and the checkpoint still has whatever
  lane-following ability it learned before collapsing -- restarting cold
  would repeat the same risky early-exploration phase that plausibly caused
  the collapse in the first place. Smoke-tested first (`--fresh`, 96
  timesteps): stall now triggers in 29 steps at -150 as expected, no errors.
  Log: `train_round10.log`. First ~11 episodes after resuming are still all
  `stall` at ~-150 (expected -- it resumed the already-collapsed policy;
  entropy needs some steps to nudge it back toward exploring driving
  behavior again). Not yet evaluated for whether the collapse resolves.
  Open question flagged but *not yet acted on*: the per-tick progress reward
  (waypoint milestone bonus + potential-based shaping) is bounded by route
  length (~60-150 waypoints per route at 2m spacing), so a policy that
  drives ~75% of a route and then crashes could still net a positive total
  reward under the current terminal penalties -- worth revisiting once
  round 10's stall-collapse fix is evaluated, but deferred for now to avoid
  changing two things at once.
  Ran the full 150k timesteps, ending at `total_timesteps=302745` (slight
  rollout-boundary overshoot past the 300,329 target). `Model saved
  successfully!`, no errors. Stall-collapse from round 9 did *not* recur as
  a pure end-state -- termination reasons in the back half of
  `episode_log.txt` are a mix of `timeout`/`stall`/`off_road`/`wrong_way`/
  `crash` rather than round 9's wall of pure `stall`, consistent with the
  entropy bonus keeping exploration alive. However, live driving via
  `drive.py` showed the car moving faster than round 9 but slower and more
  jittery than round 8 (still weaving/oscillating), and never reliably
  reaching the destination -- round 8 remains the best-driving (if
  destination-missing) checkpoint so far.

- **Round 11 (in progress)**: resumed from round 10's checkpoint (no
  `--fresh`), 150k timesteps, `--ent-coef 0.01`, targeting the persistent
  jittery/weaving "drunk walk" driving behavior:
  1. **Lane-invasion sensor gap closed**: `_on_lane_invasion()` only flagged
     crossings of `Solid`/`SolidSolid`/`SolidBroken`/`BrokenSolid` markings,
     excluding plain `Broken` (dashed) markings -- the type that separates
     most same-direction lanes. A car weaving back and forth across a dashed
     line while staying roughly on the road never triggered the `-2.0`
     lane-invasion penalty at all. Added `Broken` to the trigger set.
  2. **Lane-offset penalty strengthened**: `reward -= 0.05 * abs(lane_offset)`
     -> `0.25 * abs(lane_offset)`. At the old coefficient, even the clipped
     max offset (5m) only cost -0.25/tick, and realistic in-lane weaving cost
     next to nothing -- far too weak next to the ~0.1/tick heading bonus,
     progress shaping, and +2.0 per waypoint, so a jittery-but-progressing
     policy still netted positive reward.
  3. **Steering-smoothness penalty strengthened**: `reward -= 0.2 * abs(steer
     _action - previous_steer)` -> `0.4 * ...` -- round 9's introduction of
     this term at 0.2 was evidently insufficient to stop oscillation, per
     round 10's driving behavior.
  Smoke-tested first (`--fresh`, 96 timesteps): ran cleanly, no errors, stall
  triggered as expected. Backed up round 10's checkpoint
  (`ppo_carla_model_round10_backup.zip` /
  `ppo_carla_model_round10_backup_vecnormalize.pkl`) before resuming.

  **Switched from resume to `--fresh` partway through round 11.** Initially
  launched resumed from round 10's checkpoint, but killed it after ~69k
  steps (7 min in) once we realized resuming confounds attribution: round
  10's weaving behavior is already strongly reinforced into the policy
  weights, and PPO's clipped updates limit how fast a policy can shift per
  update, so a resumed run that still looks bad after 150k steps wouldn't
  tell us whether the round 11 reward fixes work or whether the old habit
  just hadn't been unlearned yet. A fresh run against the full current
  reward function gives a clean answer to "does this reward combination
  produce good driving from scratch." Also discovered per-step throughput is
  much higher than assumed (~500+ steps/sec, not CPU-tick-bound as earlier
  believed) now that rendering is disabled (`1980740 Disable CARLA rendering
  during training`), so a fresh 150k-step run is cheap (~5 min of raw
  stepping time) rather than the multi-hour cost assumed in earlier rounds.
  Relaunched with `--fresh --total-timesteps 150000 --ent-coef 0.01`. Log:
  `train_round11.log`.

  **Round 11 complete.** Ran the full 150k timesteps (230 episodes, ends at
  `total_timesteps=150547`), saved cleanly, no errors. Termination-reason
  breakdown over the whole run: `stall` 141, `timeout` 62, `off_road` 15,
  `wrong_way` 10, `crash` 2, `success` 0. Last 40 episodes shifted toward
  `timeout` (26) over `stall` (13) -- i.e. it moved from freezing in place
  toward surviving the full 1500 steps without crashing, similar to round
  8's failure mode (drives around, never reaches the destination).

  **Live evaluation (user, via driving the checkpoint): still bad.** Not
  moving / moving very slowly -- slower than round 10. Still visibly
  jittery/oscillating. User's assessment: every round since round 8 has made
  driving quality *worse*, not better, despite round 8 itself never reaching
  the destination. Round 8 remains the best-driving checkpoint of all 11
  rounds to date.

  This is a strong signal that the round-by-round manual reward-coefficient
  tuning approach (bump this penalty, add that bonus, re-run 150k steps,
  eyeball it) has stalled out or is actively moving in the wrong direction,
  and needs a more systematic rethink rather than another single hand-picked
  tweak -- see below for the next step taken.

- **Round 12 planning**: given round 11's regression, spawned a
  deep-reasoning research agent (Opus) to review the full round 4-11 history
  in this file plus the current `carla_gym_env.py`/`train.py` reward and
  termination logic, and propose next steps -- including whether an
  AutoML-style automated search over reward coefficients/hyperparameters
  (vs. continued hand-tuning) is viable here.

  **Key finding**: pulled the actual reward distribution per round --
  round 8 median +1168 (max +5204, 7 successes); round 9 median -217 (max
  -7.5, 0 successes); round 10 median -161 (max -14.2, 0 successes); round 11
  median -243 (**max -148.5**, 0 successes). Since round 9, *no episode out
  of 554 has earned positive total reward* -- the reward function became a
  pure cost function. Under discounted RL with an everywhere-negative
  reward, the optimal policy is to end the episode as cheaply/fast as
  possible; round 11's single best-ever outcome across 230 episodes was
  `reason=stall` at -148.5, i.e. **freezing immediately was the argmax**.
  Root causes identified (see `tutorial.md` §1.3/1.6/1.7 for the mechanics):
  1. The round 9-11 steering-smoothness penalty was computed on the
     *sampled* action, so it punished the policy's own Gaussian exploration
     noise (~-0.45/tick at init) far more strongly (~30x) than `ent_coef`
     could counteract, causing the policy to collapse its own action
     variance (stop moving) to escape the penalty.
  2. `gamma=0.99` gives only a ~100-step effective horizon (5 sec at 20Hz);
     the +500 success bonus, ~1000+ steps away, was discounted to ~0.02 --
     mathematically invisible to the agent.
  3. Round 9's obstacle-braking term (`danger * brake_action * 0.5`,
     unconditional on speed) was a second exploitable reward for not moving,
     structurally identical to round 8's original speed-reward hack.
  4. Round 11's `Broken`-marking lane-invasion penalty fired on every legal
     junction turn (dashed lines are mandatory to cross when turning),
     directly punishing the behavior most needed.
  5. Methodology: every round bundled 2-4 changes with n=1 run and no fixed
     reward-independent scoring metric, so no round's "did it help?" was
     ever actually measurable above run-to-run noise -- only eyeballing was.
  Full reasoning, code pointers, and the proposed AutoML (Optuna two-stage
  search) design are preserved in the agent's report; conceptual mechanics
  behind the diagnosis are now written up in `tutorial.md`.

- **Round 12 (in progress)**: implemented the agent's "R1 baseline + R2
  horizon fix + R3 structural jitter fix" recommendations together, fresh
  run (obs/action semantics changed enough -- action-repeat -- that
  resuming doesn't make sense):
  1. **Reward rebalanced back to net-positive for competent driving**:
     progress-shaping coefficient 0.5 -> 1.0; lane-offset penalty 0.25 ->
     0.15; terminal penalties (crash/off_road/wrong_way/stall) 150 -> 30,
     uniform; timeout 75 -> 20; success bonus stays +500. Rationale: with
     per-tick reward net-positive again, ending an episode early already
     forfeits future positive reward (discounted opportunity cost) -- a
     sufficient disincentive against dying on its own, so large terminal
     penalties stacked on top aren't needed and were actively harmful when
     per-tick reward was negative (see finding above).
  2. **Steering-smoothness reward term deleted entirely** (was fighting
     `ent_coef` for control of the policy's action variance -- see finding
     above). Replaced with two structural fixes instead:
     - **Action-repeat**: each RL decision is now held for 4 physical CARLA
       ticks (20Hz -> 5Hz effective decision rate) instead of 1 -- caps how
       often direction can change, and stretches the effective planning
       horizon 4x for free (see #3 below).
     - **Steer low-pass filter**: the *applied* steer blends 70% previous /
       30% new sample each physical tick within a repeat window, so the
       actual control signal changes smoothly even when the sampled action
       is noisy.
  3. **Gamma raised 0.99 -> 0.995** (`train.py` default), combined with
     action-repeat's 4x stretch, to make the +500 success bonus and route
     completion actually visible within the discounted objective -- chose a
     moderate increase + action-repeat over jumping straight to an extreme
     gamma (e.g. 0.9999), which would add value-function variance the
     ~150k-step budget likely can't afford (see `tutorial.md` §1.7).
  4. **Round 11's `Broken` lane-marking trigger reverted** back to
     solid-markings-only (was penalizing every legal junction turn).
  5. **Obstacle-braking reward term gated on `speed > 2.0`** so it can't be
     farmed by parking near a static prop and holding the brake indefinitely
     (was unconditional, a second exploitable "reward for not moving").
  6. **Fixed an unrelated bug found along the way**: resumed runs'
     `model.learning_rate = args.learning_rate` was a no-op (SB3 reads the
     LR from `self.lr_schedule`, not the bare attribute) -- every past
     "resumed" round silently kept its original LR schedule regardless of
     `--learning-rate`. Now uses `get_schedule_fn`.
  Not yet done (deferred, larger scope): the driving-completion sanity
  metric / expert-vs-do-nothing sanity gate (R0), CARLA server
  parallelization for multi-env training (R5), and the two-stage Optuna
  search itself -- all recommended as next steps *after* this baseline is
  confirmed to actually drive.
  CARLA's docker container had exited (code 137) between round 11 and this
  session -- restarted before smoke-testing. Smoke-tested (`--fresh`, 128
  timesteps, tiny n_steps/batch): ran cleanly, no errors, stall now costs
  ~-30 to -50 (30 terminal + small accumulated per-tick costs) as expected.
  Backed up round 11's checkpoint (`ppo_carla_model_round11_backup.zip` /
  `_vecnormalize.pkl`). Launched `--fresh --total-timesteps 150000
  --ent-coef 0.01`. Log: `train_round12.log`. Not yet evaluated.

- **Round 13 (complete)**: no source changes at all -- `git diff` against HEAD
  is empty, and the only untracked files are tooling (a `start_tensorboard.sh`
  launcher, an opencode hooks config, and the `AGENTS.md` -> `CLAUDE.md`
  symlink). The round is purely a continuation: loaded round 12's checkpoint
  (`fresh=False`, no code or hyperparameter change) and ran another ~150k
  timesteps -- the log's first episode starts at `total_timesteps=151525`
  (round 12 ended at 151,525) and the run ends at `total_timesteps=302983`,
  with the same hyperparameters as round 12 (gamma 0.995, ent_coef 0.01,
  n_steps 2048, lr 3e-4). Log: `train_round13.log`.

  **This is the first round that reliably reaches the destination.** 1,557
  episodes; termination-reason breakdown: `crash` 538, `off_road` 416,
  `wrong_way` 298, `stall` 90, `timeout` 8, **`success` 307 (19.7%)** in
  rollout episodes vs. 43 successes total in round 12's 5,296 episodes. Mean
  episode reward +234.8
  (max +1298.96 on a 368-step success; successes average +848.8 over 147
  steps). Successes climbed steadily through the round -- 35 / 49 / 96 / 127
  per quartile, and 68 (34%) in the last 200 episodes -- whereas round 12's
  final ~116 episodes had only 5. The in-training periodic deterministic eval
  tracked the same trend (40/140 successes overall, 39/119 = 32.8% in the
  back half). **Standalone `eval.py` on the final checkpoint: 6/10 successes**
  (user-reported, deterministic).

  **Interpretation**: the only thing that changed between rounds 12 and 13 was
  ~150k more timesteps of the identical recipe -- this is the strongest
  evidence yet that round 12's redesign (net-positive per-tick reward,
  action-repeat + steer low-pass, gamma 0.995, smoothness-penalty deletion,
  `Broken`-marking revert, speed-gated obstacle braking) fixed the structural
  problems that kept rounds 9-11 regressing, rather than the improvement being
  another luck-of-the-n=1-tuning artifact. The car went from "best-driving but
  never-completes" (round 8) to actually finishing routes once the recipe got
  enough steps. The success rate was still rising when the run ended (no
  plateau), so the checkpoint likely isn't saturated yet.

  **Caveats / open questions** (don't over-read the 6/10): (a) n=1 run, and
  6/10 on 10 fresh attempts is a noisy estimate of true completion rate
  (~±15%); (b) attempts get a random route, so coverage over the harder
  junction-heavy route variants is unknown; (c) training data shows the failure
  modes are far from solved -- 538 crashes, 416 off-road, 298 wrong-way over
  the round, and the periodic eval still occasionally hits `stall` (~-30), so
  the degenerate no-move mode is reachable but no longer dominant; (d)
  `ppo_carla_model.zip` / `_vecnormalize.pkl` now hold the round-13 model;
  round 12's exact checkpoint is preserved as
  `ppo_carla_model_round12_backup.zip` / `_vecnormalize.pkl` (backed up at
  round 13 launch). The previously deferred items from round 12's plan (R0
  expert/do-nothing sanity baseline, parallelized multi-env training, Optuna
  search) are the natural next steps now that there's a baseline that actually
  completes routes.

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
