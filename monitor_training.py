#!/usr/bin/env python3
"""Monitor training progress and report statistics."""

import time
import os
from collections import defaultdict

def parse_episode_log(filename):
    """Parse episode log and extract metrics."""
    if not os.path.exists(filename):
        return None

    episodes = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Episode' in line and 'reward=' in line:
                try:
                    parts = line.split('Episode ')[1].split(': ')
                    ep_num = int(parts[0])

                    reward_str = line.split('reward=')[1].split(',')[0]
                    reward = float(reward_str)

                    length_str = line.split('length=')[1].split(' ')[0]
                    length = int(length_str)

                    timesteps_str = line.split('total_timesteps=')[1].strip()
                    timesteps = int(timesteps_str)

                    episodes.append({
                        'num': ep_num,
                        'reward': reward,
                        'length': length,
                        'timesteps': timesteps
                    })
                except:
                    pass

    return episodes

def print_progress(episodes):
    """Print training progress statistics."""
    if not episodes:
        print("No episodes logged yet...")
        return

    recent = episodes[-100:]  # Last 100 episodes
    rewards = [e['reward'] for e in recent]
    lengths = [e['length'] for e in recent]

    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(lengths) / len(lengths)
    max_reward = max(rewards)
    min_reward = min(rewards)
    total_steps = episodes[-1]['timesteps']

    print("\n" + "="*70)
    print(f"Training Progress: {len(episodes)} episodes, {total_steps} timesteps")
    print("="*70)
    print(f"Avg Reward (last 100):  {avg_reward:8.2f}  (min={min_reward:8.2f}, max={max_reward:8.2f})")
    print(f"Avg Episode Length:     {avg_length:8.1f} steps")
    print(f"Latest Episode:         #{episodes[-1]['num']} reward={episodes[-1]['reward']:8.2f}, length={episodes[-1]['length']} steps")

    # Trend
    if len(episodes) > 200:
        old_avg = sum([e['reward'] for e in episodes[-200:-100]]) / 100
        new_avg = sum([e['reward'] for e in episodes[-100:]]) / 100
        trend = "📈 UP" if new_avg > old_avg else "📉 DOWN"
        print(f"Trend (last 200 eps):    {trend} ({old_avg:.2f} → {new_avg:.2f})")

    print("="*70)

def main():
    """Monitor training in a loop. Stops when the train.py process (tracked
    via train.pid) is no longer running, rather than assuming a fixed
    timestep target -- this also works when continuing from a checkpoint
    whose timestep counter starts above the run's total_timesteps."""
    log_file = "episode_log.txt"
    pid_file = "train.pid"
    interval = 30  # Check every 30 seconds

    print(f"Starting training monitor. Checking {log_file} every {interval}s...")

    last_ep_count = 0
    while True:
        try:
            episodes = parse_episode_log(log_file)

            if episodes and len(episodes) > last_ep_count:
                print_progress(episodes)
                last_ep_count = len(episodes)

            # Check if the training process has exited
            if os.path.exists(pid_file):
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                if not os.path.exists(f"/proc/{pid}"):
                    print("\n✓ Training process has exited.")
                    if episodes:
                        print_progress(episodes)
                    break

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n✓ Monitor stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    main()
