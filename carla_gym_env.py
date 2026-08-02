import gymnasium as gym
from gymnasium import spaces
import carla
import numpy as np
import random
import time
import math
import sys
import os

# The GlobalRoutePlanner lives in <carla repo>/PythonAPI/carla/agents, alongside
# (not inside) the installed `carla` pip package. Adding that directory to the
# path makes `agents` importable as a top-level package. CARLA_ROOT should
# point at the root of the CARLA repo checkout (defaults to a sibling
# checkout at ~/git/carla, since this project lives outside that repo).
CARLA_ROOT = os.environ.get("CARLA_ROOT", os.path.expanduser("~/git/carla"))
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
from agents.navigation.global_route_planner import GlobalRoutePlanner

class CarlaGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CarlaGymEnv, self).__init__()

        # Connect to your Dockerized CARLA Server
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # Enable synchronous mode for deterministic training/eval
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)

        # Route planner: resolves an actual point-A-to-point-B path along the road graph
        self.route_planner = GlobalRoutePlanner(self.map, sampling_resolution=2.0)

        self.actor_list = []
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.crashed = False
        self.off_road = False
        self.lane_invaded_this_step = False
        self.waypoints = []  # Current route waypoints
        self.waypoint_index = 0
        self.goal_location = None
        self.episode_steps = 0
        self.max_episode_steps = 1500  # ~75 seconds at 20 Hz
        self.stall_counter = 0
        self.prev_distance_to_goal = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.waypoint_lookahead = 5  # Average heading/curvature over next N waypoints

        # 1. Define Action Space: Continuous values for [Steering (-1.0 to 1.0), Throttle (0.0 to 1.0)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # 2. Define Observation Space: Real values
        # [Forward Speed (km/h), Distance to next waypoint (m),
        #  Angle diff averaged over next N waypoints (rad), Distance to final goal (m),
        #  Lateral offset from lane center (m), Heading error vs. road direction (rad),
        #  Previous steer action, Previous throttle action]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -5.0, -np.pi, -1.0, 0.0]),
            high=np.array([50.0, 200.0, np.pi, 500.0, 5.0, np.pi, 1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()  # Wipe out old actors from previous episodes

        # A destroyed actor's collision footprint isn't actually cleared
        # until the next tick -- without this, spawning at the same
        # fixed_start transform used by the previous episode reliably fails
        # with a collision RuntimeError, silently falling back to a random
        # spawn point below (defeating the whole point of a fixed start/goal
        # repeat, e.g. drive.py's multi-attempt comparison on one route).
        self.world.tick()

        # Allow a fixed start/goal to be requested (e.g. to repeat the same
        # attempt multiple times). options = {"start_transform": carla.Transform,
        # "goal_location": carla.Location}
        options = options or {}
        fixed_start = options.get("start_transform")
        fixed_goal = options.get("goal_location")

        blueprint = self.blueprint_library.filter("model3")[0]  # Tesla Model 3
        self.vehicle = None
        start_location = None

        if fixed_start is not None:
            try:
                # fixed_start is usually a *resting* transform captured from
                # a previously-settled vehicle (e.g. drive.py repeating the
                # same route), whose z is a hair below true ground level from
                # suspension compression -- spawning a fresh vehicle at that
                # exact height collides with the static road mesh itself, not
                # just a leftover actor, so lift it slightly and let the
                # existing settle-tick loop below drop it back down.
                lifted_start = carla.Transform(
                    carla.Location(fixed_start.location.x, fixed_start.location.y,
                                   fixed_start.location.z + 0.5),
                    fixed_start.rotation
                )
                self.vehicle = self.world.spawn_actor(blueprint, lifted_start)
                start_location = fixed_start.location
            except RuntimeError as e:
                # Falls through to a random spawn below -- surfaced loudly
                # since silently landing on a different start than requested
                # defeats the purpose of asking for a fixed one.
                print(f"! Failed to spawn at requested fixed_start ({fixed_start.location}): "
                      f"{e} -- falling back to a random spawn point.")
                self.vehicle = None

        if self.vehicle is None:
            # Random spawn (with retry logic for collision at spawn point)
            spawn_points = self.map.get_spawn_points()
            random.shuffle(spawn_points)
            for spawn_point in spawn_points:
                try:
                    self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
                    start_location = spawn_point.location
                    break
                except RuntimeError:
                    continue

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at any spawn point. Is CARLA running? Are all spawn points occupied?")

        self.actor_list.append(self.vehicle)

        # Attach collision sensor
        self.crashed = False
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.collision_sensor)

        # Attach lane invasion sensor (fires on crossing a lane marking)
        self.off_road = False
        self.lane_invaded_this_step = False
        lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))
        self.actor_list.append(self.lane_invasion_sensor)

        # Generate the route: fixed goal if provided, otherwise a random one
        self._generate_route(start_location, fixed_goal=fixed_goal)

        # Wrong-way detection itself happens per-step in step() by comparing
        # the vehicle's heading to its *current* lane's local direction (see
        # the comment there) -- nothing to precompute here beyond resetting
        # the flag.
        self.wrong_way = False

        # Reset episode step counter
        self.episode_steps = 0
        self.stall_counter = 0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0

        # Let the car drop and settle safely (tick 10 times instead of sleeping)
        for _ in range(10):
            self.world.tick()

        # Extract initial state observation
        obs = self._get_observation()
        self.prev_distance_to_goal = float(obs[3])
        info = {}
        return obs, info

    def step(self, action):
        # 1. Apply actions predicted by Stable-Baselines3
        steer_action = float(action[0])
        throttle_action = float(action[1])

        control = carla.VehicleControl(throttle=throttle_action, steer=steer_action, brake=0.0)
        self.vehicle.apply_control(control)

        # Reset the per-step lane invasion flag before ticking -- the sensor
        # callback will set it if a solid marking is crossed during this tick.
        self.lane_invaded_this_step = False

        # Synchronous tick
        self.world.tick()
        self.episode_steps += 1

        # Off-road check: project_to_road=False returns None when the vehicle's
        # location isn't inside any drivable lane (e.g. drove off the road
        # entirely, onto grass/sidewalk with no lane marking to trigger the
        # lane invasion sensor).
        current_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=False)
        self.off_road = current_waypoint is None

        # Track the action that produced this new state, so it shows up as
        # "previous action" in the observation computed below.
        self.prev_steer = steer_action
        self.prev_throttle = throttle_action

        # 2. Extract new observation vectors
        obs = self._get_observation()

        # Wrong-way check: compare the vehicle's heading to its *current*
        # lane's own local direction (obs[5], from _get_observation()) --
        # more than 90 degrees off means driving against that lane's traffic
        # flow. This replaced an earlier version that compared the current
        # lane's OpenDrive lane_id sign against the sign recorded once at
        # reset(): that sign convention is only consistent within a single
        # road segment, not across the whole route, so turning onto a
        # different road at an intersection could flip it and falsely flag
        # a perfectly correct turn as wrong-way -- this per-step, per-lane
        # check has no such cross-road dependency.
        self.wrong_way = (not self.off_road) and abs(obs[5]) > (np.pi / 2)
        if self.wrong_way:
            debug_wp = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
            vehicle_yaw = self.vehicle.get_transform().rotation.yaw
            print(f"! wrong_way triggered: road_id={debug_wp.road_id if debug_wp else 'N/A'}, "
                  f"lane_id={debug_wp.lane_id if debug_wp else 'N/A'}, "
                  f"is_junction={debug_wp.is_junction if debug_wp else 'N/A'}, "
                  f"lane_yaw={debug_wp.transform.rotation.yaw if debug_wp else float('nan'):.1f}, "
                  f"vehicle_yaw={vehicle_yaw:.1f}, "
                  f"heading_error_deg={math.degrees(obs[5]):.1f}")

        # 3. Reward function: incentivize speed + following waypoints, penalize crashes
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

        reward = speed * 0.1  # Base reward for moving
        reward -= 0.01  # Small per-step penalty to encourage efficiency

        # Penalty for idling (not moving AND not accelerating)
        if speed < 1.0 and throttle_action < 0.1:
            reward -= 0.3  # Mild penalty for laziness

        # Potential-based shaping on distance-to-goal: telescopes to a total
        # bounded by net distance closed over the episode, independent of
        # route length -- unlike a flat per-waypoint bonus, this can't dwarf
        # the terminal crash/off-road penalties on long routes.
        distance_to_goal = obs[3]
        reward += 0.5 * (self.prev_distance_to_goal - distance_to_goal)
        self.prev_distance_to_goal = float(distance_to_goal)

        # Advance waypoint if we're close enough (only if actually moving).
        # Small fixed bonus kept as a milestone signal, not the main reward driver.
        if len(self.waypoints) > self.waypoint_index:
            distance_to_waypoint = obs[1]
            if distance_to_waypoint < 2.0 and speed > 2.0:  # Within 2m AND moving at least 2 km/h
                self.waypoint_index += 1
                reward += 2.0

        # Reward for heading toward waypoint (only if moving)
        if speed > 1.0:  # Only reward steering when actually moving
            angle_error = abs(obs[2])
            reward += max(0, 0.5 - angle_error / np.pi) * 0.5  # Reduced bonus

        # Continuous penalty for drifting off lane center / misaligning with
        # road heading, instead of relying solely on the sparse lane-invasion
        # event -- gives a per-tick gradient toward staying centered.
        lane_offset = obs[4]
        heading_error = obs[5]
        reward -= 0.05 * abs(lane_offset)
        reward -= 0.1 * abs(heading_error)

        # Penalize crossing a solid lane marking (wrong-lane / off-road edge)
        if self.lane_invaded_this_step:
            reward -= 2.0

        terminated = False
        truncated = False
        termination_reason = None

        # 4. Check terminal criteria
        # Track sustained stalling (full throttle, not moving) rather than a
        # single-step snapshot -- right after reset/spawn, speed is legitimately
        # near 0 for the first few ticks while the car is still accelerating
        # from a standstill, so a one-step check falsely flags that as "stuck".
        if speed < 0.1 and self.vehicle.get_control().throttle > 0.5:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        if self.crashed:
            reward -= 100.0
            terminated = True
            termination_reason = "crash"
        elif self.off_road:
            reward -= 75.0
            terminated = True
            termination_reason = "off_road"
        elif self.wrong_way:
            reward -= 75.0
            terminated = True
            termination_reason = "wrong_way"
        elif self.stall_counter >= 20:  # ~1 second of sustained stalling
            reward -= 10.0
            terminated = True
            termination_reason = "stall"
        elif self.waypoint_index >= len(self.waypoints):
            # Reached the end of the route
            reward += 20.0
            terminated = True
            termination_reason = "success"

        # 5. Episode truncation (time limit) -- penalize running out the clock
        # without reaching the destination, otherwise a policy that loops in
        # place collecting small per-tick rewards forever has no incentive to
        # actually finish the route.
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
            reward -= 50.0
            termination_reason = "timeout"

        return obs, reward, terminated, truncated, {"termination_reason": termination_reason}

    def _generate_route(self, start_location, min_manhattan_distance=100.0, max_manhattan_distance=200.0, fixed_goal=None):
        """Pick a goal within [min, max] Manhattan distance and resolve a real route to it.
        If fixed_goal is given, use it directly instead of picking randomly."""
        if fixed_goal is not None:
            self.goal_location = fixed_goal
            route = self.route_planner.trace_route(start_location, self.goal_location)
            self.waypoints = [wp for wp, _ in route] if route else [self.map.get_waypoint(start_location)]
            self.waypoint_index = 0
            return

        spawn_points = self.map.get_spawn_points()

        def in_range(sp, lo, hi):
            manhattan = (abs(sp.location.x - start_location.x) +
                         abs(sp.location.y - start_location.y))
            return lo <= manhattan <= hi

        candidates = [sp for sp in spawn_points if in_range(sp, min_manhattan_distance, max_manhattan_distance)]

        # Widen the search if nothing matched (sparse maps / unlucky start point)
        lo, hi = min_manhattan_distance, max_manhattan_distance
        while not candidates and hi < min_manhattan_distance * 10:
            lo = max(0.0, lo - 30.0)
            hi = hi + 30.0
            candidates = [sp for sp in spawn_points if in_range(sp, lo, hi)]

        goal_spawn = random.choice(candidates) if candidates else random.choice(spawn_points)
        self.goal_location = goal_spawn.location

        # Resolve an actual road-graph route from start to goal
        route = self.route_planner.trace_route(start_location, self.goal_location)
        self.waypoints = [wp for wp, _ in route] if route else [self.map.get_waypoint(start_location)]
        self.waypoint_index = 0

    def _get_observation(self):
        """Calculate real observations: speed, distance/angle to next waypoint, distance to goal."""
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

        distance_to_waypoint = 0.0
        angle_to_waypoint = 0.0
        vehicle_loc = self.vehicle.get_location()
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw

        # Distance to the immediate next waypoint (used to trigger waypoint
        # advancement -- that needs the precise next point, not an average).
        if len(self.waypoints) > self.waypoint_index:
            waypoint_loc = self.waypoints[self.waypoint_index].transform.location
            dx = waypoint_loc.x - vehicle_loc.x
            dy = waypoint_loc.y - vehicle_loc.y
            distance_to_waypoint = np.sqrt(dx**2 + dy**2)

        # Heading error averaged (circular mean) over the next few waypoints
        # instead of just the single next one -- a lone waypoint's placement
        # jitter can look like a steering-relevant angle change on a straight
        # road when it isn't; averaging over a short lookahead smooths that
        # noise out while still tracking upcoming curvature.
        lookahead = self.waypoints[self.waypoint_index:self.waypoint_index + self.waypoint_lookahead]
        if lookahead:
            sin_sum = 0.0
            cos_sum = 0.0
            for wp in lookahead:
                wp_loc = wp.transform.location
                wdx = wp_loc.x - vehicle_loc.x
                wdy = wp_loc.y - vehicle_loc.y
                wp_yaw = np.arctan2(wdy, wdx) * 180 / np.pi
                angle_diff = wp_yaw - vehicle_yaw
                angle_diff = ((angle_diff + 180) % 360) - 180  # Normalize
                angle_rad = angle_diff * np.pi / 180
                sin_sum += math.sin(angle_rad)
                cos_sum += math.cos(angle_rad)
            angle_to_waypoint = math.atan2(sin_sum, cos_sum)

        # Distance to the final destination (helps the critic value states by
        # "how far from done", independent of next-waypoint noise)
        distance_to_goal = 0.0
        if self.goal_location is not None:
            gdx = self.goal_location.x - vehicle_loc.x
            gdy = self.goal_location.y - vehicle_loc.y
            distance_to_goal = np.sqrt(gdx**2 + gdy**2)

        # Lane-center offset and road-heading alignment, computed from the
        # *current* lane waypoint (not the route waypoint list) so it's a
        # stable, always-available signal even between sparse route waypoints.
        lane_offset = 0.0
        heading_error = 0.0
        current_lane_wp = self.map.get_waypoint(vehicle_loc, project_to_road=True)
        if current_lane_wp is not None:
            lane_tf = current_lane_wp.transform
            lane_loc = lane_tf.location
            lane_yaw_rad = math.radians(lane_tf.rotation.yaw)

            # Signed lateral distance: project (vehicle - lane_center) onto
            # the lane's right vector.
            ldx = vehicle_loc.x - lane_loc.x
            ldy = vehicle_loc.y - lane_loc.y
            right_x = math.sin(lane_yaw_rad)
            right_y = -math.cos(lane_yaw_rad)
            lane_offset = ldx * right_x + ldy * right_y
            lane_offset = float(np.clip(lane_offset, -5.0, 5.0))

            heading_diff = vehicle_yaw - lane_tf.rotation.yaw
            heading_diff = ((heading_diff + 180) % 360) - 180  # Normalize to [-180, 180]
            heading_error = heading_diff * np.pi / 180

        return np.array([speed, distance_to_waypoint, angle_to_waypoint, distance_to_goal,
                          lane_offset, heading_error, self.prev_steer, self.prev_throttle],
                         dtype=np.float32)

    def _on_collision(self, event):
        self.crashed = True

    def _on_lane_invasion(self, event):
        for marking in event.crossed_lane_markings:
            if marking.type in (carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid,
                                 carla.LaneMarkingType.SolidBroken, carla.LaneMarkingType.BrokenSolid):
                self.lane_invaded_this_step = True

    def _cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []
        self.collision_sensor = None
        self.lane_invasion_sensor = None

    def close(self):
        self._cleanup()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)