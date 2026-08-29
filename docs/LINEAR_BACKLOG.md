# Linear setup and initial backlog

This document is an execution plan, not a substitute for `THINKING.md` or
`PROGRESS.md`.

## Project

- Name: `CARLA RL Driving`
- Purpose: turn the current prototype into a reproducible research platform,
  then test increasingly difficult driving hypotheses one at a time.

## Workflow

`Backlog -> Ready -> Implementing -> Running -> Analyzing -> Done`

An experiment may move to **Ready** only after the researcher has written its
observation, hypothesis, confidence, intervention, prediction, and proposed
measurement in `THINKING.md`. It moves to **Done** only after results and
unresolved questions have been appended to `PROGRESS.md`.

## Labels

- Kind: `infrastructure`, `experiment`, `bug`, `learning`
- Area: `simulator`, `environment`, `evaluation`, `algorithm`, `perception`
- Risk: `behavior-preserving`, `changes-observation`, `changes-reward`,
  `changes-dynamics`, `confound-risk`

## Milestone 1: Reproducible baseline

1. **Freeze the pre-speed-limit baseline code and artifacts**
   - Record source revision, artifact checksums, observation/action shape, and
     embedded algorithm metadata.
   - Acceptance: a manifest identifies the exact code/checkpoint pair without
     relying on a filename such as `ppo_carla_model`.
2. **Add versioned server and experiment configuration**
   - Acceptance: one resolved configuration captures simulator, environment,
     algorithm, evaluation, and seed settings.
3. **Create immutable run manifests**
   - Acceptance: every run records Git SHA/dirty state, dependencies, config
     fingerprint, server slot, and source diff before training begins.
4. **Pin and document the runtime environment**
   - Acceptance: CARLA/Python/SB3/PyTorch/Gymnasium versions can be recreated.
5. **Add characterization tests for the current environment**
   - Observation names/order/ranges, reward components, action transformation,
     and termination priority.

## Milestone 2: Simulator isolation

6. **Replace `.bashrc` CARLA aliases with scoped repository orchestration**
   - Three independent Docker services; stop operations affect only this project.
7. **Add exclusive simulator-slot leasing**
   - A training, evaluation, or driving client cannot accidentally share a
     synchronous world with another ticking client.
8. **Move evaluation to a dedicated simulator**
   - Remove same-training-environment resets from `PeriodicEvalCallback`.
9. **Add a one-versus-three simulator throughput benchmark**
   - Report environment steps/second, GPU memory, wall clock per PPO update,
     and rollout size. Treat `n_steps * n_envs` as an explicit algorithm change.

## Milestone 3: Fixed-route evaluation

10. **Define fixed train/validation/test route IDs and seeds**
11. **Make Gymnasium seeding control route and spawn sampling**
12. **Implement reward-independent driving metrics**
    - Route completion fraction, infractions/km, termination reason,
      steering rate, lateral acceleration, jerk, and elapsed simulation time.
13. **Evaluate stationary, random, CARLA BasicAgent, and PPO baselines**
14. **Promote or reject `state-v1-round16b` as the validated baseline**

## Milestone 4: Cross-town generalization

15. **Author the cross-town hypothesis in `THINKING.md`**
16. **Evaluate the frozen policy on two held-out towns without retraining**
17. **Analyze failures by road topology and junction type**
18. **Decide whether multi-town training is warranted**

## Later milestones

19. Quantify and localize steering jitter on fixed routes.
20. Test one-sided speed-limit compliance as an isolated reward experiment.
21. Add lead-vehicle following.
22. Add traffic-light stopping and restart.
23. Add cross-traffic negotiation.
24. Add pedestrian emergency-braking scenarios.
25. Compare PPO and SAC under identical scenario, seed, step, and wall-clock budgets.
26. Build a privileged-state-to-camera perception teacher task.
27. Replace one privileged observation group with learned perception.
28. Consider end-to-end visual fine-tuning only after perception and control are
    independently measurable.

## Experiment issue template

```markdown
## Research question

## THINKING.md reference

## Single hypothesis

## Intervention and control

## Predicted outcomes

## Fixed scenario suite and seeds

## Reward-independent primary metric

## Confound audit

## Run IDs / Git revisions

## Result and decision

## PROGRESS.md reference
```

