"""
Evaluate the trained model across multiple episodes.
Collects metrics: mean/std reward, success rate (episodes without crashes).
"""

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from carla_gym_env import CarlaGymEnv
import numpy as np

def evaluate_model(model, env, n_episodes=10):
    """
    Custom evaluation: run episodes and track crashes + rewards.
    """
    episode_rewards = []
    episode_lengths = []
    crashes = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Rough heuristic: if episode was very short, likely a crash
        if episode_length < 50:
            crashes += 1

        print(f"Episode {episode + 1:2d}: reward={episode_reward:7.2f}, steps={episode_length:4d}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = (n_episodes - crashes) / n_episodes

    print(f"\n{'='*50}")
    print(f"Mean Reward:     {mean_reward:7.2f} ± {std_reward:7.2f}")
    print(f"Mean Episode Length: {mean_length:7.1f} steps")
    print(f"Success Rate:    {success_rate*100:5.1f}% ({n_episodes - crashes}/{n_episodes} episodes)")
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

    print("Running 10 evaluation episodes...")
    metrics = evaluate_model(model, env, n_episodes=10)

    env.close()

if __name__ == "__main__":
    main()
