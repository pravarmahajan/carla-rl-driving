import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from carla_gym_env import CarlaGymEnv
import os

class PeriodicSaveCallback(BaseCallback):
    """Save to the canonical model path periodically during a long run, so
    drive.py always has access to the most recent checkpoint instead of only
    whatever was saved at the end of the previous completed run."""
    def __init__(self, model_path, vecnormalize_path, save_freq=10000):
        super().__init__()
        self.model_path = model_path
        self.vecnormalize_path = vecnormalize_path
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(self.model_path)
            self.training_env.save(self.vecnormalize_path)
            self._last_save = self.num_timesteps
        return True


class PeriodicEvalCallback(BaseCallback):
    """Run deterministic evaluation episodes on the *same* env/vehicle used
    for training, at each rollout boundary. A truly separate eval env would
    need its own CARLA vehicle in the same synchronous world -- but CARLA's
    world.tick() advances every actor at once regardless of which client
    issued it, so a second env ticking the world mid-rollout would drag the
    idle training vehicle along with it (uncontrolled drift/crashes) and
    desync its episode-step bookkeeping. Reusing the single training env
    avoids that: this callback pauses collection, runs full eval episodes to
    completion, then hands control back with fresh state.

    Runs at rollout boundaries (not every step) so it never has to interrupt
    a training episode mid-flight -- collect_rollouts() only calls
    _on_rollout_end() between complete rollouts, never inside one.
    """
    def __init__(self, eval_freq_rollouts=5, n_eval_episodes=10):
        super().__init__()
        self.eval_freq_rollouts = eval_freq_rollouts
        self.n_eval_episodes = n_eval_episodes
        self._rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.eval_freq_rollouts != 0:
            return

        venv = self.training_env  # VecNormalize-wrapped DummyVecEnv, n_envs=1
        venv.training = False  # freeze obs/reward running stats during eval

        episode_rewards = []
        episode_lengths = []
        successes = 0
        squared_value_errors = []
        reason_counts = {}

        for _ in range(self.n_eval_episodes):
            episode_reward = 0.0
            episode_length = 0
            step_values = []
            step_rewards = []
            termination_reason = None

            # Must reset before *every* eval episode: the previous episode
            # left the env in a needs_reset state, and Monitor raises
            # RuntimeError on step() otherwise. This also draws a fresh
            # random spawn/route per episode, so eval episodes are i.i.d.
            # samples of the route distribution rather than continuations.
            obs = venv.reset()
            while True:
                obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    value_pred = self.model.policy.predict_values(obs_tensor).cpu().numpy().flatten()[0]
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = venv.step(action)
                # Un-normalized reward for the human-comparable eval/mean_reward
                # metric -- VecNormalize's step() return value is reward-normalized.
                reward = venv.get_original_reward()[0]
                # value_pred above came from the critic, which was trained to
                # predict returns on VecNormalize's *normalized* reward scale
                # (norm_reward=True) -- comparing it against a return built
                # from raw rewards was comparing different units entirely
                # (e.g. eval/value_loss reading ~1e5 while train/value_loss
                # read ~0.1). Normalize each reward the same way VecNormalize
                # does before accumulating the return, so both sides of the
                # squared-error comparison are in the scale the critic
                # actually learned.
                normalized_reward = float(venv.normalize_reward(np.array([reward]))[0])
                step_values.append(value_pred)
                step_rewards.append(normalized_reward)
                episode_reward += reward
                episode_length += 1
                if dones[0]:
                    termination_reason = infos[0].get("termination_reason", "unknown")
                    if termination_reason == "success":
                        successes += 1
                    break
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            reason_counts[termination_reason] = reason_counts.get(termination_reason, 0) + 1
            print(f"Eval ep {self._rollout_count}.{_ + 1}: reward={episode_reward:.2f}, "
                  f"length={episode_length}, reason={termination_reason}")

            # Empirical discounted return realized from each step to the end
            # of this (deterministic, held-out) episode, compared against
            # what the critic predicted for that step at the time -- a
            # value-loss signal computed off-policy from eval data, distinct
            # from train/value_loss (fit only to noisy on-policy GAE targets
            # from the training rollout). Treats truncation the same as
            # termination (no bootstrapped tail) -- a reasonable
            # approximation for a diagnostic, not used for any gradient.
            returns = np.zeros(len(step_rewards))
            running_return = 0.0
            for t in reversed(range(len(step_rewards))):
                running_return = step_rewards[t] + self.model.gamma * running_return
                returns[t] = running_return
            squared_value_errors.extend((np.array(step_values) - returns) ** 2)

        self.logger.record("eval/mean_reward", float(np.mean(episode_rewards)))
        self.logger.record("eval/mean_length", float(np.mean(episode_lengths)))
        self.logger.record("eval/success_rate", successes / self.n_eval_episodes)
        self.logger.record("eval/n_success", successes)
        for reason in sorted(reason_counts):
            self.logger.record(f"eval/n_{reason}", reason_counts[reason])
        self.logger.record("eval/value_loss", float(np.mean(squared_value_errors)))
        self.logger.dump(self.num_timesteps)

        venv.training = True
        # We've been driving the vec env directly (outside collect_rollouts),
        # so the algorithm's cached "last obs" is now stale. Reset and hand
        # back fresh state -- otherwise the next rollout would start from an
        # observation that no longer matches the env's actual state.
        self.model._last_obs = venv.reset()
        self.model._last_episode_starts = np.ones((venv.num_envs,), dtype=bool)


class EpisodeLoggerCallback(BaseCallback):
    """Log metrics only at the end of each episode."""
    def __init__(self, log_file="episode_log.txt"):
        super().__init__()
        self.episode_count = 0
        self.log_file = log_file
        with open(self.log_file, "w") as f:
            f.write("Episode Logger Started\n")

    def _on_step(self) -> bool:
        # Monitor only injects the "episode" key into info on the exact step
        # an episode actually ends -- unlike ep_info_buffer, which stays
        # non-empty forever once the first episode completes.
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            self.episode_count += 1
            reason = info.get("termination_reason", "unknown")

            log_msg = (f"Episode {self.episode_count}: "
                      f"reward={episode_info['r']:.2f}, "
                      f"length={episode_info['l']} steps, "
                      f"reason={reason}, "
                      f"total_timesteps={self.num_timesteps}\n")

            with open(self.log_file, "a") as f:
                f.write(log_msg)

            # Log to TensorBoard with episode number as global step
            self.logger.record("episode/reward", episode_info['r'], exclude="stdout")
            self.logger.record("episode/length", episode_info['l'], exclude="stdout")
            self.logger.record("episode/number", self.episode_count, exclude="stdout")

            print(log_msg.strip())

        return True

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent to drive in CARLA")
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout length per update")
    parser.add_argument("--n-epochs", type=int, default=10, help="Gradient epochs per rollout")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995,
                         help="Discount factor (round 12: 0.99 -> 0.995 -- combined with "
                              "action_repeat=4 in carla_gym_env.py, extends PPO's effective "
                              "horizon enough for the +500 success bonus to actually be visible)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss weight")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy bonus weight (SB3 default is 0)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                         help="Number of deterministic eval episodes per eval point (was 3 -- "
                              "too few to resolve a ~3% success rate)")
    parser.add_argument("--model-path", type=str, default="ppo_carla_model",
                         help="Path (without .zip) to load an existing model from and save back to")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing model at --model-path and start from scratch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Define a state-saving directory for tensorboard logs
    logdir = "./logs/"
    os.makedirs(logdir, exist_ok=True)

    vecnormalize_path = args.model_path + "_vecnormalize.pkl"

    model_zip = args.model_path + ".zip"
    resume = not args.fresh and os.path.exists(model_zip)

    # VecNormalize needs a VecEnv underneath it -- DummyVecEnv with a single
    # env is the standard way to get one out of a plain gym.Env. Monitor must
    # wrap the env *before* DummyVecEnv: SB3 only auto-adds it when you hand
    # PPO a raw gym.Env, not when you hand it an already-vectorized VecEnv
    # (which is what we're doing here to get VecNormalize in front of it) --
    # without it, no "episode" info key ever gets set, so EpisodeLoggerCallback
    # and SB3's own rollout/ep_rew_mean stats silently never fire.
    venv = DummyVecEnv([lambda: Monitor(CarlaGymEnv(no_rendering=True))])

    if resume and os.path.exists(vecnormalize_path):
        print(f"Loading existing observation/reward normalization stats from {vecnormalize_path}...")
        venv = VecNormalize.load(vecnormalize_path, venv)
    else:
        # norm_reward clips/normalizes the reward *fed to PPO's advantage
        # estimation*, not the raw reward logged in episode_log.txt/TensorBoard
        # (those come from the original, unnormalized `info["episode"]`).
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)

    if resume:
        print(f"Loading existing model from {model_zip} to continue training...")
        model = PPO.load(args.model_path, env=venv, tensorboard_log=logdir)
        # Hyperparameters below only take effect for a fresh model; loaded
        # models keep whatever they were originally trained with unless
        # explicitly overridden here.
        # SB3 reads the LR from self.lr_schedule (a callable), not from
        # self.learning_rate directly -- setting the latter alone here was a
        # no-op that silently kept every resumed run on its original
        # schedule regardless of --learning-rate. get_schedule_fn wraps a
        # constant into the callable form SB3's optimizer actually consults.
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
    else:
        print("Starting a fresh model.")
        # "MlpPolicy" tells SB3 to look at your vector space (instead of image/CNN pixels)
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=0,
            tensorboard_log=logdir,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
        )

    print("--- Starting Reinforcement Learning Pipeline ---")
    print(f"Hyperparameters: {vars(args)}")

    callback = CallbackList([
        EpisodeLoggerCallback(),
        PeriodicSaveCallback(args.model_path, vecnormalize_path, save_freq=10000),
        PeriodicEvalCallback(eval_freq_rollouts=5, n_eval_episodes=args.eval_episodes),
    ])
    model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)

    # Save your trained neural weights + the matching normalization stats
    # (drive.py/eval.py need both to reproduce the same obs distribution the
    # policy was trained on).
    model.save(args.model_path)
    venv.save(vecnormalize_path)
    print("Model saved successfully!")

    venv.close()