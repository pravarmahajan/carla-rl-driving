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
import argparse
import json
from pathlib import Path

from carla_rl.launch import DEFAULT_SERVER_CATALOG, create_configured_run


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
    termination_counts = {}
    episodes = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        termination_reason = "unknown"

        while True:
            predict_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(predict_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)
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
        termination_counts[termination_reason] = termination_counts.get(termination_reason, 0) + 1
        episodes.append(
            {
                "episode": episode + 1,
                "reward": float(episode_reward),
                "steps": episode_length,
                "termination_reason": termination_reason,
            }
        )

        print(f"Episode {episode + 1:2d}: reward={episode_reward:7.2f}, "
              f"steps={episode_length:4d}, outcome={termination_reason}")

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    success_rate = float((n_episodes - failures) / n_episodes)

    print(f"\n{'='*50}")
    print(f"Mean Reward:     {mean_reward:7.2f} ± {std_reward:7.2f}")
    print(f"Mean Episode Length: {mean_length:7.1f} steps")
    print(f"Success Rate:    {success_rate*100:5.1f}% ({n_episodes - failures}/{n_episodes} episodes)")
    print(f"{'='*50}")

    return {
        "n_episodes": n_episodes,
        "n_success": n_episodes - failures,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "termination_counts": termination_counts,
        "episodes": episodes,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CARLA model")
    parser.add_argument("-p", "--port", type=int, default=2000)
    parser.add_argument("--model-path", default="ppo_carla_model",
                        help="Checkpoint path without .zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--config", type=Path,
                        help="Versioned experiment config. Creates an immutable evaluation run record.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_SERVER_CATALOG)
    parser.add_argument("--server", help="Server slot override used with --config")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_dir = None
    env_kwargs = {"no_rendering": True, "port": args.port}
    if args.config:
        resolved, run_dir = create_configured_run(
            args.config, args.catalog, args.server, args.runs_root, args.run_name, "evaluate",
            runtime_overrides={"model_path": args.model_path, "episodes": args.episodes},
        )
        environment = resolved["environment"]
        simulator = resolved["simulator"]
        env_kwargs = {
            "no_rendering": simulator["no_rendering"],
            "host": simulator["host"],
            "port": simulator["rpc_port"],
            "fixed_delta_seconds": simulator["fixed_delta_seconds"],
            "action_repeat": environment["action_repeat"],
            "steer_lowpass_alpha": environment["steer_lowpass_alpha"],
            "max_physical_ticks": environment["max_physical_ticks"],
            "seed": resolved["algorithm"]["seed"],
            "town": simulator["town"],
        }
        print(f"Created reproducible evaluation run: {run_dir}")

    env = CarlaGymEnv(**env_kwargs)
    model = PPO.load(args.model_path, env=env)
    print(f"✓ Model loaded: {args.model_path}\n")

    normalizer = load_normalizer(env, args.model_path + "_vecnormalize.pkl")
    print("✓ Loaded observation normalization stats" if normalizer is not None
          else "! No normalization stats found -- assuming an older, unnormalized model\n")

    print(f"Running {args.episodes} evaluation episodes...")
    metrics = evaluate_model(model, env, normalizer=normalizer, n_episodes=args.episodes)
    if run_dir is not None:
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    env.close()

if __name__ == "__main__":
    main()
