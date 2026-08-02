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
import math
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


class MiniMap:
    """Top-down route map (route line, start, goal, live vehicle position +
    heading) drawn in a corner of the window. The front-facing camera alone
    gives no sense of where the destination is or how much route remains --
    this answers "is it actually heading toward the goal" at a glance."""
    def __init__(self, waypoints, goal_location, start_location, size=240, margin=14):
        self.size = size
        self.margin = margin
        self.route_points = [(wp.transform.location.x, wp.transform.location.y) for wp in waypoints]
        self.start = (start_location.x, start_location.y)
        self.goal = (goal_location.x, goal_location.y)

        xs = [p[0] for p in self.route_points] + [self.start[0], self.goal[0]]
        ys = [p[1] for p in self.route_points] + [self.start[1], self.goal[1]]
        pad = 10.0
        self.min_x, self.max_x = min(xs) - pad, max(xs) + pad
        self.min_y, self.max_y = min(ys) - pad, max(ys) + pad

    def _to_px(self, x, y):
        span_x = max(self.max_x - self.min_x, 1e-3)
        span_y = max(self.max_y - self.min_y, 1e-3)
        scale = min((self.size - 2 * self.margin) / span_x, (self.size - 2 * self.margin) / span_y)
        px = self.margin + (x - self.min_x) * scale
        py = self.margin + (y - self.min_y) * scale
        return int(px), int(py)

    def render(self, display, vehicle_transform, top_left):
        surface = pygame.Surface((self.size, self.size))
        surface.fill((25, 25, 25))

        route_px = [self._to_px(x, y) for x, y in self.route_points]
        if len(route_px) > 1:
            pygame.draw.lines(surface, (110, 110, 110), False, route_px, 2)

        sx, sy = self._to_px(*self.start)
        pygame.draw.circle(surface, (80, 160, 255), (sx, sy), 4)

        gx, gy = self._to_px(*self.goal)
        pygame.draw.circle(surface, (230, 50, 50), (gx, gy), 6)
        pygame.draw.circle(surface, (255, 255, 255), (gx, gy), 6, 1)

        vx, vy = self._to_px(vehicle_transform.location.x, vehicle_transform.location.y)
        yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        tip = (vx + 9 * math.cos(yaw_rad), vy + 9 * math.sin(yaw_rad))
        left = (vx + 5 * math.cos(yaw_rad + 2.5), vy + 5 * math.sin(yaw_rad + 2.5))
        right = (vx + 5 * math.cos(yaw_rad - 2.5), vy + 5 * math.sin(yaw_rad - 2.5))
        pygame.draw.polygon(surface, (255, 220, 0), [tip, left, right])

        pygame.draw.rect(surface, (200, 200, 200), surface.get_rect(), 1)
        display.blit(surface, top_left)


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
        # Fresh random start/goal each attempt, so we see behavior across a
        # variety of roads/turns instead of repeating one identical scenario
        # (the policy + environment are deterministic, so re-running the same
        # fixed route just reproduces the same trajectory every time).
        obs, _ = env.reset()
        vehicle = env.vehicle
        minimap = MiniMap(env.waypoints, env.goal_location, vehicle.get_transform().location)

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

            minimap.render(display, vehicle.get_transform(), top_left=(800 - 240 - 10, 10))

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
    print("Summary over 3 attempts (each on a different random route):")
    for i, (steps, reward, outcome) in enumerate(results):
        print(f"  Attempt {i + 1}: {steps:4d} steps, reward={reward:8.2f}, {outcome}")
    print("=" * 60)

if __name__ == "__main__":
    main()
