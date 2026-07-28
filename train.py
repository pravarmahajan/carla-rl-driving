import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from carla_gym_env import CarlaGymEnv
import os

class PeriodicSaveCallback(BaseCallback):
    """Save to the canonical model path periodically during a long run, so
    drive.py always has access to the most recent checkpoint instead of only
    whatever was saved at the end of the previous completed run."""
    def __init__(self, model_path, save_freq=10000):
        super().__init__()
        self.model_path = model_path
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(self.model_path)
            self._last_save = self.num_timesteps
        return True


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

            log_msg = (f"Episode {self.episode_count}: "
                      f"reward={episode_info['r']:.2f}, "
                      f"length={episode_info['l']} steps, "
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
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss weight")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy bonus weight (SB3 default is 0)")
    parser.add_argument("--model-path", type=str, default="ppo_carla_model",
                         help="Path (without .zip) to load an existing model from and save back to")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing model at --model-path and start from scratch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create the customized environment
    env = CarlaGymEnv()

    # Define a state-saving directory for tensorboard logs
    logdir = "./logs/"
    os.makedirs(logdir, exist_ok=True)

    model_zip = args.model_path + ".zip"
    if not args.fresh and os.path.exists(model_zip):
        print(f"Loading existing model from {model_zip} to continue training...")
        model = PPO.load(args.model_path, env=env, tensorboard_log=logdir)
        # Hyperparameters below only take effect for a fresh model; loaded
        # models keep whatever they were originally trained with unless
        # explicitly overridden here.
        model.learning_rate = args.learning_rate
    else:
        print("Starting a fresh model.")
        # "MlpPolicy" tells SB3 to look at your vector space (instead of image/CNN pixels)
        model = PPO(
            "MlpPolicy",
            env,
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
        PeriodicSaveCallback(args.model_path, save_freq=10000),
    ])
    model.learn(total_timesteps=args.total_timesteps, callback=callback, reset_num_timesteps=False)

    # Save your trained neural weights
    model.save(args.model_path)
    print("Model saved successfully!")

    env.close()