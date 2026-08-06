# Tutorial: concepts explained along the way

This is a companion to `PROGRESS.md`. `PROGRESS.md` tracks *what changed and
what happened* round by round. This file collects the *conceptual*
explanations that came up when digging into why -- PPO mechanics, reward
design principles, and some infra/tooling questions. Organized by topic, not
chronologically.

---

## 1. PPO / RL fundamentals

### 1.1 How does PPO training actually work, mechanically?

1. **Rollout**: for `n_steps` env steps, the current policy network takes a
   state and outputs a mean action `μ(s)`, samples an actual action
   `a ~ N(μ(s), σ)`, and that sampled `a` is what's actually sent to CARLA.
   The env returns a reward.
2. **Advantage estimation**: after the rollout, PPO computes an *advantage*
   `A` per (state, action) pair via GAE -- roughly "was this action better or
   worse than expected from this state?", using the value network.
3. **Gradient update**: PPO's loss is
   ```
   loss = -clipped_surrogate(ratio, A) + vf_coef * value_loss - ent_coef * entropy
   ```
   where `ratio = π_new(a|s) / π_old(a|s)`. This is backpropagated through
   the network via ordinary gradient descent -- same as any DL training loop.

The core intuition: **whatever action got sampled and turned out to have
high advantage gets pushed toward higher probability; low/negative advantage
gets pushed down.**

### 1.2 The policy's action distribution: where does sigma (σ) come from?

For continuous actions, SB3's policy outputs a Gaussian per action
dimension: a mean `μ(s)` **and** a standard deviation `σ`. It's easy to
assume `σ` is just another raw output of the neural network like `μ` is --
it isn't. `μ(s)` is state-dependent, computed by a real forward pass through
trained weights. **`σ` is a separate, free-standing trainable parameter**
(`log_std`, one scalar per action dimension) that does **not** depend on the
input state at all. It's initialized via a hyperparameter (`log_std_init`,
default `0.0`), and since `σ = exp(log_std)`, `exp(0) = 1.0` -- that's why
`σ` starts at 1.0. It's a deliberate library default, not an emergent
property of network initialization.

### 1.3 How does gradient descent shrink sigma?

`log_std` gets two competing gradient pressures every update:

- **From the policy-gradient term**: if large sampled deviations in some
  action dimension keep landing on low/negative advantage, gradient descent
  reduces the probability of sampling large deviations in the future -- for
  a Gaussian, that means shrinking `σ`.
- **From the entropy bonus**: Gaussian entropy has a closed form,
  `H = 0.5 * log(2πe·σ²)` per dimension -- monotonically increasing in `σ`.
  The `-ent_coef * entropy` loss term means gradient descent pushes `σ` *up*
  (more entropy -> lower loss, weighted by `ent_coef`).

Whichever pressure is numerically larger wins. In this project, the round
9-11 steering-smoothness reward term (`-0.4 * |steer_action - previous_steer|`,
computed on the *sampled* action) created a pressure to shrink steering's
`σ` roughly **30x stronger** than `ent_coef=0.01`'s pressure to keep it up --
because at `σ≈1`, two consecutive independent Gaussian samples differ by
~1.13 on average, so the smoothness penalty was largely punishing the
policy's own exploration noise, not "real" erratic driving. The policy's
only way out was to collapse `σ` (stop exploring), which looks like "give up
and barely move." See `PROGRESS.md` round 12 for the fix (action-repeat +
low-pass filtering instead of a reward penalty).

### 1.4 Is sigma shared across all action dimensions (steer/throttle/brake)?

No -- SB3's default gives **one independent `log_std` parameter per action
dimension**. Steering's `σ` and throttle's `σ` are mechanically separate
numbers; shrinking one doesn't *directly* shrink the other.

But there's still real coupling, through a different path: **the advantage
`A` for a transition is a single scalar covering the whole joint
(steer, throttle, brake) action**, not computed per-dimension (since
`log π(a|s) = log π(steer) + log π(throttle) + log π(brake)`, and the loss
scales that whole sum by the same `A`). So if a transition's advantage is
strongly negative mainly because of a steering penalty, the gradient for
*all three* dimensions' log-probs on that transition still gets scaled down
together -- whatever throttle/brake values happened to co-occur with a
heavily-penalized steering sample get pushed down in probability too, by
association. There's also a softer coupling: all three action means
typically share the same hidden-layer trunk, so if the value function learns
"the safest state is not moving," that pressure shows up in the mean output
for every dimension, not just steering.

### 1.5 Where does `ent_coef` come from, and is it part of the environment?

It's a pure PPO/algorithm hyperparameter (`train.py --ent-coef`, SB3 default
`0.0`) -- nothing to do with CARLA or `carla_gym_env.py` at all. It's a
straight scalar weight in the loss function (see 1.3). Rounds up to and
including round 9 never passed this flag (so it was silently `0.0` the whole
time -- no entropy bonus at all, nothing pushing exploration back once the
policy converged onto a low-variance behavior). Round 10 was the first round
to pass `--ent-coef 0.01`.

### 1.6 How does negative per-tick reward + discounting punish "dying"? (and why doesn't it, currently)

This principle applies to a **healthy, mostly-positive-per-tick reward**,
not to what rounds 9-11 actually had.

**Under a positive-leaning reward**: every additional tick spent driving
well earns more reward. Crashing at tick 200 instead of continuing to tick
1500 means *forfeiting* all the reward that would have accrued in the
remaining 1300 ticks -- a real opportunity cost. Discounting (`γ<1`) shrinks
that lost future reward a little the further out it is, but it's still
substantial. So ending an episode early is already expensive by default,
without needing a huge explicit terminal penalty on top.

**Under a negative-per-tick reward** (what rounds 9-11 actually had): the
logic inverts. Every additional tick *costs* more. So ending the episode
early *saves* you from future losses -- exactly backwards. Stacking a large
terminal penalty (-150) on top of that doesn't fix it, because the real
problem is that *surviving itself* is the thing being penalized. This is why
round 12 both reduced the terminal penalties **and** rebalanced the per-tick
terms to be net-positive for competent driving -- doing only one of those
would still leave a broken incentive.

### 1.7 Why does `gamma` (discount factor) matter here, and should it be ~0.9999?

Effective planning horizon ≈ `1 / (1 - γ)`, in units of *decisions*:
- `γ=0.99` → 100-decision horizon.
- `γ=0.9999` → 10,000-decision horizon.

At 20 Hz with one decision per physical tick, `γ=0.99` means the agent is
effectively optimizing "what happens in the next 5 seconds" -- a goal 1500
ticks away is invisible; a success bonus that far out is discounted by
`0.99^1500 ≈ 0`, worth less than a single tick's lane-offset penalty.

But jumping straight to `γ=0.9999` isn't free: it makes the value function's
target much noisier (predicting returns over a much longer, more uncertain
window), and typically needs far more training data to converge reliably.
On a single-environment, ~150k-step budget, an extreme jump risks trading
"can't see the goal" for "value estimates are too noisy to learn anything."

Round 12's approach: two moderate levers together instead of one extreme
one.
- `γ: 0.99 → 0.995` (500-decision horizon) -- a meaningful increase without
  destabilizing the value function too much.
- **Action-repeat** (hold each action for 4 physical ticks, see below):
  since decisions now happen every 4 ticks instead of every 1, the same
  `γ=0.995` horizon of ~500 *decisions* now covers ~2000 physical ticks --
  four times the real-time horizon, "for free," without paying γ's
  variance cost at all.

If the goal is still effectively invisible after this, nudge `γ` up further
from there and re-measure -- start moderate, don't reach for the extreme
value first.

### 1.8 What are action low-pass filtering and action-repeat?

Both fix "jittery/erratic steering" at the *control* layer instead of the
*reward* layer, so you don't need a reward term that fights the entropy
bonus (see 1.3/1.6).

**Action low-pass (smoothing filter)**: instead of applying the policy's raw
sampled action directly, blend it with the previously *applied* action:
```python
applied_steer = 0.7 * previous_applied_steer + 0.3 * raw_sampled_steer
```
An exponential moving average. Even if the policy's raw output is noisy
tick-to-tick, the actual physical steering command changes smoothly, because
each new command is mostly the old one plus a small nudge. The noise gets
damped in actuation, not punished in reward -- entropy is left alone.

**Action-repeat (frame-skip)**: instead of the policy choosing a new action
every single 20 Hz tick, it chooses one action every N ticks (round 12 uses
N=4, i.e. 5 Hz effective decision rate) and that action is held for all N
physical ticks. This structurally caps how often direction *can* change --
there's no way to oscillate faster than the decision rate allows -- while
CARLA still simulates physics at the finer tick rate internally. It's a
standard technique for exactly this kind of stochastic-policy-on-continuous-
control twitchiness, and (per 1.7) it also extends the effective time
horizon for free.

They're complementary: action-repeat caps *how often* direction can change;
low-pass smooths *how much* it changes when it does.

---

## 2. Reward design principles

### 2.1 Is there a penalty for lane crossing? Does weaving evade it?

Two separate mechanisms existed, and originally both were weak/gapped:

1. **Event-based** (`_on_lane_invasion`): only fires on *solid* lane
   markings (`Solid`/`SolidSolid`/`SolidBroken`/`BrokenSolid`). Plain
   `Broken` (dashed) markings -- what separates most same-direction lanes --
   were excluded, so weaving back and forth across a dashed line never
   triggered it at all. Round 11 added `Broken` to the trigger set to close
   this gap -- but that also meant every legal turn through a junction
   (which requires crossing dashed connector-lane markings) started
   incurring the penalty, directly punishing turning. Round 12 reverted this
   -- see `PROGRESS.md` round 12.
2. **Continuous** (`lane_offset` in the reward function): a per-tick penalty
   proportional to lateral distance from lane center, independent of marking
   type. Originally weighted so weakly (`0.05`) that realistic in-lane
   weaving cost almost nothing next to other per-tick bonuses.

### 2.2 AutoML in an RL context -- how is it different from DL hyperparameter search?

| | DL AutoML | RL AutoML |
|---|---|---|
| **Cost of one trial** | One run to a loss curve on a fixed dataset -- cheap, fast, i.i.d. batches | One full training run of an agent interacting with an environment -- expensive, sequential, no fixed dataset |
| **What's being optimized** | A fixed, given loss function | Often *both* algorithm hyperparameters *and* the reward function itself (a design choice, not a given) |
| **How you measure a trial** | Validation loss on held-out data -- cheap, reusable | Roll out the trained policy across multiple episodes/seeds to get a noisy estimate of quality -- evaluation itself costs simulation time |
| **Noise per trial** | Modest (init seed) | High -- both training (exploration randomness) and evaluation (stochastic policy + stochastic env) are noisy |

The search algorithms themselves are similar in spirit:

- **Optuna-style (Bayesian/TPE search + pruning)**: propose a hyperparameter
  combo, run a full (or partial) training run, report a score back, update
  the search's belief, propose the next combo. The RL-specific addition is
  **pruning**: since a full run is expensive, check in periodically during
  training (e.g. every 10k steps) and kill clearly-underperforming trials
  early, rather than always running to completion.
- **Population-Based Training (PBT)**: run several agents in parallel with
  different hyperparameters. Periodically, weaker agents copy the weights
  *and* hyperparameters of stronger ones ("exploit"), then randomly perturb
  the copied hyperparameters ("explore") and keep training. This handles
  something DL search usually doesn't need to: hyperparameters that should
  *change over training* (e.g. more exploration early, less later) -- PBT
  discovers a schedule rather than a single fixed value.
- **Critical RL-specific trap**: the scoring metric must be independent of
  whatever's being searched. If a reward-coefficient search is scored on the
  shaped reward it's tuning, it will simply find coefficients that trivially
  inflate that reward (i.e. rediscover reward hacking) rather than producing
  good driving. Score trials on a fixed, hand-defined, reward-independent
  metric instead (e.g. route-completion % + infraction count from a
  deterministic rollout on fixed seeded routes).

---

## 3. Project infra / tooling questions

### 3.1 Why are episodes slow? Does adding more CPU cores help?

CARLA's synchronous-mode stepping (`world.tick()`) is an inherently serial,
single-threaded-per-environment RPC loop between the Python client and the
CARLA server process. Both `CarlaUE4` and `train.py` were observed pinned at
~100% of exactly *one* core each (`ps`/`top`), regardless of how many cores
the machine has (confirmed 24-28 available) -- the bottleneck isn't compute
parallelism, it's the serial tick round-trip. More idle cores don't speed up
a single environment; true throughput gains require running multiple
parallel CARLA server instances (e.g. `SubprocVecEnv` with one CARLA server
per env, each on a different RPC port).

Separately: per-step throughput turned out to be much higher than assumed
once CARLA's client-side rendering was disabled during training (commit
"Disable CARLA rendering during training") -- observed ~500+ steps/sec
afterward, vs. the multi-hour-per-150k-steps cost assumed in earlier rounds.

### 3.2 Does *resuming* from a checkpoint make training slower than starting fresh?

No -- checked empirically. Model/VecNormalize loading is a one-time cost at
startup, not a per-step cost; the forward pass through this small MLP policy
costs the same whether the weights were just initialized or just loaded from
disk, and per-step time is dominated by the CARLA tick round-trip either
way. The perceived slowdown is more likely because a *better* (less
immediately-crashing) resumed policy runs longer episodes (up to the
1500-step timeout) instead of ending in a 30-100 step stall/crash --
fewer *episodes* complete per minute even though timesteps/sec is
unchanged, which can look like "slower" if you're watching episode count.

### 3.3 Resume vs. fresh -- which should we use when testing a reward change?

Resuming carries forward whatever behavior is already strongly reinforced
into the policy's weights. If that behavior is a bug you're trying to fix
(e.g. round 10/11's weaving), a resumed run under a new reward has to first
*unlearn* the old habit before the new reward's effect is even visible --
and PPO's clipped updates deliberately limit how fast a policy can shift per
update, so unlearning a well-reinforced habit can take a long time. A run
that still looks bad after 150k resumed steps doesn't tell you whether the
new reward doesn't work, or whether it just hasn't finished unlearning yet --
that's a confound, not a clean signal.

A **fresh** run against a reward function answers a cleaner question: "does
this reward, learned from scratch, produce good behavior?" It's the right
choice specifically when you're trying to evaluate whether a *reward design*
works, as opposed to continuing to improve an already-good policy. (Round 11
switched from resume to fresh mid-run for exactly this reason.)

### 3.4 What does `episode/number` mean in TensorBoard, and why did it jump oddly?

It's just the running count of completed episodes within *that specific
training process*, logged as a scalar against `total_timesteps` on the
x-axis (`train.py`'s `EpisodeLoggerCallback`). The odd non-monotonic
jumps/resets seen on the chart were a plotting artifact, not a real
training signal: `logs/PPO_1` has accumulated event files from **every**
training process launched since round 5 (fresh runs, resumed runs, and
throwaway smoke tests), all sharing the same TensorBoard run name, so their
step axes overlap/interleave in the merged chart. Fix (not yet applied):
pass a distinct `tb_log_name` per round to `model.learn(...)` so future
charts aren't merged across historical runs.

### 3.5 SB3 resume timestep semantics

For a resumed model, `model.learn(total_timesteps=X, reset_num_timesteps=False)`
is **additive**: training runs until `self.num_timesteps` reaches
`(timesteps_at_resume + X)`, not just X more from zero. The stop condition
is only checked at rollout boundaries (`n_steps` granularity), so the actual
final `total_timesteps` typically overshoots the target slightly.
