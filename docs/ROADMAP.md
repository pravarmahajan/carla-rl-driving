# Research roadmap

## Decision

Do not begin end-to-end vision training yet. First make the existing state-vector
policy reproducible and measurable, then evaluate it unchanged on held-out towns.

## Stage 0: Foundation

- Isolate CARLA servers and clients.
- Capture complete run manifests.
- Separate training and evaluation worlds.
- Establish deterministic scenario selection and behavior-preserving tests.

Exit criterion: another engineer can reproduce the same fixed-route evaluation
from a run ID without asking which branch, model file, port, or command was used.

## Stage 1: Generalization before complexity

Run the frozen baseline on two held-out towns without retraining. This tests the
claim that route-relative state features transfer across road geometry.

Exit criterion: route-completion and infraction confidence intervals are
reported by town and topology, with failures linked to replayable scenario IDs.

## Stage 2: Control quality

Measure raw policy steer, applied steer, yaw rate, lateral acceleration, and
jerk on fixed routes. Distinguish policy-mean oscillation from exploration,
filtering, and vehicle-dynamics effects before changing action repeat again.

## Stage 3: Interactive driving

Add one capability per experiment: one-sided speed compliance, lead-vehicle
following, traffic lights, cross traffic, then pedestrians. Prefer scripted,
repeatable scenarios before randomized traffic density.

## Stage 4: Algorithm comparison

Compare PPO and SAC only after scenario distribution and metrics are fixed. Use
multiple seeds, equal simulator steps, and report wall-clock cost and simulator
throughput alongside driving metrics.

## Stage 5: Perception

Keep proprioceptive speed and a high-level route command. First use simulator
state as labels for a camera perception model, validate perception separately,
then replace one privileged feature group in the controller. Pixels-only PPO
from scratch combines too many unmeasured changes for the next experiment.

