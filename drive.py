"""
Load the trained model and drive it in the simulator with camera visualization.
Displays the vehicle's camera feed in a pygame window.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from carla_gym_env import CarlaGymEnv
import carla
import pygame
import numpy as np
import weakref
import os
import datetime


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

class CameraCapture:
    """Captures camera images from the vehicle and stores latest frame."""
    def __init__(self, vehicle, world, width=800, height=600):
        self.vehicle = vehicle
        self.world = world
        self.width = width
        self.height = height
        self.image = None
        self.camera = None

        # Attach RGB camera to vehicle
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "110")

        # Position camera on vehicle (front-facing)
        transform = carla.Transform(
            carla.Location(x=0.5, z=1.5),
            carla.Rotation(pitch=0)
        )
        self.camera = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.camera.listen(lambda img: self._on_image(weakref.ref(self), img))

    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        # Convert to RGB and store
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        self.image = array[:, :, :3]  # Drop alpha channel

    def get_image(self):
        """Return latest captured image (RGB numpy array)."""
        return self.image

    def destroy(self):
        if self.camera and self.camera.is_alive:
            self.camera.destroy()


def render_to_pygame(image, display):
    """Convert numpy RGB image to pygame surface and render."""
    if image is None:
        return
    # Swap axes for pygame: (H, W, 3) -> (W, H, 3)
    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    display.blit(surface, (0, 0))


def main():
    # Initialize environment
    env = CarlaGymEnv()
    world = env.world

    # Load the trained model
    model_path = "ppo_carla_model"
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(model_path + ".zip"))
    model = PPO.load(model_path, env=env)
    print(f"✓ Model loaded: {model_path}.zip (saved {mtime:%Y-%m-%d %H:%M:%S})")

    normalizer = load_normalizer(env, model_path + "_vecnormalize.pkl")
    print("✓ Loaded observation normalization stats" if normalizer is not None
          else "! No normalization stats found -- assuming an older, unnormalized model")

    # Pick ONE start/goal pair by doing a throwaway reset, then reuse it for
    # every attempt so all 3 tries are on the identical route.
    env.reset()
    start_transform = env.vehicle.get_transform()
    goal_location = env.goal_location
    print(f"Route fixed for this run: start={start_transform.location}, goal={goal_location}")

    reset_options = {"start_transform": start_transform, "goal_location": goal_location}

    # Initialize pygame
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA Agent Driving")
    clock = pygame.time.Clock()

    camera = None
    num_attempts = 3
    results = []

    for attempt in range(num_attempts):
        print(f"\n--- Attempt {attempt + 1}/{num_attempts} ---")
        obs, _ = env.reset(options=reset_options)
        vehicle = env.vehicle

        # Attach camera to newly spawned vehicle
        if camera is not None:
            camera.destroy()
        camera = CameraCapture(vehicle, world, width=800, height=600)

        episode_reward = 0.0
        episode_steps = 0
        reached_goal = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt()

            # Get action from trained policy
            predict_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(predict_obs, deterministic=True)

            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Render camera feed and HUD
            display.fill((0, 0, 0))
            image = camera.get_image()
            render_to_pygame(image, display)

            # Draw HUD text
            font = pygame.font.Font(None, 36)
            text = font.render(
                f"Attempt {attempt + 1} | Step {episode_steps} | Reward: {episode_reward:.1f}",
                True, (255, 255, 255)
            )
            display.blit(text, (10, 10))

            pygame.display.flip()
            clock.tick(20)  # 20 FPS for visibility

            episode_reward += reward
            episode_steps += 1

            # Print status every 50 steps
            if episode_steps % 50 == 0:
                print(f"  Step {episode_steps}: reward={episode_reward:.2f}")

            if terminated or truncated:
                # Reached the goal iff all waypoints were consumed without crashing/stalling
                reached_goal = (not env.crashed) and env.waypoint_index >= len(env.waypoints)
                break

        outcome = "REACHED GOAL" if reached_goal else ("CRASHED" if env.crashed else "DID NOT FINISH")
        print(f"✓ Attempt {attempt + 1} complete: {episode_steps} steps, "
              f"reward={episode_reward:.2f}, outcome={outcome}")
        results.append((episode_steps, episode_reward, outcome))

    # Cleanup
    if camera is not None:
        camera.destroy()
    env.close()
    pygame.quit()

    print("\n" + "=" * 60)
    print("Summary over 3 attempts on the same route:")
    for i, (steps, reward, outcome) in enumerate(results):
        print(f"  Attempt {i + 1}: {steps:4d} steps, reward={reward:8.2f}, {outcome}")
    print("=" * 60)

if __name__ == "__main__":
    main()
