"""
Evaluate the trained model across multiple episodes.
Collects metrics: mean/std reward, success rate (episodes without crashes).
"""

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from carla_gym_env import CarlaGymEnv
import numpy as np
import os


def load_normalizer(env, vecnormalize_path):
    """Load the obs normalization stats saved alongside a trained model, so
    inference sees the same normalized observation distribution the policy
    was trained on. Wraps the *existing* env instance (not a fresh one) --
    we only use the returned object's normalize_obs(), never step/reset it,
    so this doesn't touch CARLA or spawn anything extra."""
    if not os.path.exists(vecnormalize_path):
        return None
    dummy = DummyVecEnv([lambda: env])
    return VecNormalize.load(vecnormalize_path, dummy)


def evaluate_model(model, env, normalizer=None, n_episodes=10):
    """
    Custom evaluation: run episodes and track rewards + outcomes.
    """
    episode_rewards = []
    episode_lengths = []
    failures = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        termination_reason = "unknown"

        while True:
            predict_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(predict_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                # Read the exact cause from the env, same as drive.py does --
                # instead of guessing from episode length (a short episode is
                # not necessarily a crash, and a long one is not necessarily
                # clean).
                termination_reason = info.get("termination_reason", "unknown")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Match train.py's PeriodicEvalCallback semantics: any outcome other
        # than "success" (crash/off_road/wrong_way/stall/timeout) is a failure.
        if termination_reason != "success":
            failures += 1

        print(f"Episode {episode + 1:2d}: reward={episode_reward:7.2f}, "
              f"steps={episode_length:4d}, outcome={termination_reason}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = (n_episodes - failures) / n_episodes

    print(f"\n{'='*50}")
    print(f"Mean Reward:     {mean_reward:7.2f} ± {std_reward:7.2f}")
    print(f"Mean Episode Length: {mean_length:7.1f} steps")
    print(f"Success Rate:    {success_rate*100:5.1f}% ({n_episodes - failures}/{n_episodes} episodes)")
    print(f"{'='*50}")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "success_rate": success_rate,
    }

def main():
    env = CarlaGymEnv()
    model = PPO.load("ppo_carla_model", env=env)
    print("✓ Model loaded: ppo_carla_model\n")

    normalizer = load_normalizer(env, "ppo_carla_model_vecnormalize.pkl")
    print("✓ Loaded observation normalization stats" if normalizer is not None
          else "! No normalization stats found -- assuming an older, unnormalized model\n")

    print("Running 10 evaluation episodes...")
    metrics = evaluate_model(model, env, normalizer=normalizer, n_episodes=10)

    env.close()

if __name__ == "__main__":
    main()
