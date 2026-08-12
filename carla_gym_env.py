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

    def __init__(self, no_rendering=False):
        super(CarlaGymEnv, self).__init__()
        self.no_rendering = no_rendering

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
        # no_rendering_mode is a server-wide setting, not per-client -- always
        # set it explicitly (rather than leaving whatever a previous session
        # left behind) so there's no cross-run leakage. Training never
        # attaches a camera (only event-based collision/lane-invasion
        # sensors), so disabling rendering entirely there skips CARLA's
        # render-thread work per tick; drive.py needs it on for its camera
        # feed, so it keeps the default (no_rendering=False).
        settings.no_rendering_mode = no_rendering
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

        # Round 12: each RL decision is now held for `action_repeat` physical
        # CARLA ticks (20 Hz -> 5 Hz effective decision rate) instead of one.
        # This structurally caps how often steering can change direction (the
        # policy literally cannot whip the wheel every 50ms anymore), and
        # extends PPO's effective time horizon (gamma^n_decisions covers
        # action_repeat times more real seconds) without needing an extreme
        # gamma value. Steering is additionally low-pass filtered tick-to-tick
        # within each repeat window so the *applied* control changes smoothly
        # even though the sampled action itself is noisy -- this replaces the
        # old reward-based steering-smoothness penalty (which was taxing the
        # policy's own Gaussian exploration noise and fighting ent_coef; see
        # PROGRESS.md round 12).
        self.action_repeat = 4
        self.steer_lowpass_alpha = 0.3  # applied = 0.7*prev + 0.3*raw each tick

        # 1. Define Action Space: Continuous values for [Steering (-1.0 to
        # 1.0), Throttle (0.0 to 1.0), Brake (0.0 to 1.0)]. Brake used to be
        # hardcoded to 0.0 in step() -- the agent had no way to slow down for
        # an obstacle, only steer/accelerate.
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.obstacle_lookahead = 30.0  # meters; also the "nothing detected" sentinel value

        # 2. Define Observation Space: Real values
        # [Forward Speed (km/h), Distance to next waypoint (m),
        #  Angle diff averaged over next N waypoints (rad), Distance to final goal (m),
        #  Lateral offset from lane center (m), Heading error vs. road direction (rad),
        #  Previous steer action, Previous throttle action,
        #  Distance to nearest obstacle ahead (m, capped/sentinel at obstacle_lookahead)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -5.0, -np.pi, -1.0, 0.0, 0.0]),
            high=np.array([50.0, 200.0, np.pi, 500.0, 5.0, np.pi, 1.0, 1.0, self.obstacle_lookahead]),
            dtype=np.float32
        )

        # Static level geometry (e.g. parked-car props baked into the map)
        # doesn't move and isn't a CARLA Actor, so it never shows up in
        # world.get_actors() -- it has to be queried separately via
        # get_level_bbs(). Cached once at construction since it's fixed for
        # the lifetime of the loaded map.
        self.static_vehicle_bboxes = list(self.world.get_level_bbs(carla.CityObjectLabel.Car))

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
        # 1. Unpack the RL action once -- held fixed (steer additionally
        # low-pass filtered tick-to-tick, see __init__) across
        # self.action_repeat physical CARLA ticks below. Round 12: this
        # replaces the old single-tick-per-decision loop.
        steer_action = float(action[0])
        throttle_action = float(action[1])
        brake_action = float(action[2])

        total_reward = 0.0
        terminated = False
        truncated = False
        termination_reason = None
        obs = None

        for _ in range(self.action_repeat):
            # Low-pass filter the *applied* steer toward the sampled action
            # instead of jumping straight to it -- smooths out the physical
            # control even when the sampled action itself is noisy (see
            # __init__ comment). throttle/brake are applied as-is.
            applied_steer = self.prev_steer + self.steer_lowpass_alpha * (steer_action - self.prev_steer)

            control = carla.VehicleControl(throttle=throttle_action, steer=applied_steer, brake=brake_action)
            self.vehicle.apply_control(control)

            # Reset the per-tick lane invasion flag before ticking -- the
            # sensor callback will set it if a marking is crossed this tick.
            self.lane_invaded_this_step = False

            # Synchronous tick
            self.world.tick()
            self.episode_steps += 1

            # Off-road check: project_to_road=False returns None when the
            # vehicle's location isn't inside any drivable lane (e.g. drove
            # off the road entirely, onto grass/sidewalk with no lane
            # marking to trigger the lane invasion sensor).
            current_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=False)
            # Junctions are carved out of the off-road test (mirrors the
            # wrong_way check below): inside a junction the "drivable" area
            # is defined by connector-lane polygons that don't tile the whole
            # paved junction surface, so a normal, legal turn can momentarily
            # put the vehicle on junction asphalt that no connector polygon
            # covers -- get_waypoint(project_to_road=False) returns None
            # there even though nothing is wrong. This was terminating valid
            # episodes mid-turn. Only call it off_road when the *projected*
            # lane waypoint is a normal road segment. Tradeoff: a genuinely
            # off-road excursion that stays within a junction's bounds won't
            # terminate until the vehicle exits onto a regular road segment
            # (lane-invasion sensor still catches marking crossings meanwhile).
            current_lane_wp = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
            in_junction = current_lane_wp.is_junction if current_lane_wp else False
            self.off_road = (current_waypoint is None) and (not in_junction)

            # Track the action that produced this new state, so it shows up
            # as "previous action" in the observation computed below, and so
            # the low-pass filter above blends from the actually-applied
            # value next tick.
            self.prev_steer = applied_steer
            self.prev_throttle = throttle_action

            # 2. Extract new observation vectors
            obs = self._get_observation()

            # Wrong-way check: compare the vehicle's heading to its *current*
            # lane's own local direction (obs[5], from _get_observation()) --
            # more than 90 degrees off means driving against that lane's
            # traffic flow. Skipped entirely inside junctions: a junction's
            # connector lanes curve rapidly and the projected lane's heading
            # swings through the turn, so a normal, correct turn can
            # transiently look >90 degrees off even though nothing is wrong.
            # off_road detection and the lane-invasion sensor still catch bad
            # driving once the vehicle exits back onto a normal road segment.
            # (current_lane_wp / in_junction are computed in the off-road
            # check above and reused here -- one get_waypoint call per tick.)
            self.wrong_way = (not self.off_road) and (not in_junction) and abs(obs[5]) > (np.pi / 2)
            if self.wrong_way:
                vehicle_yaw = self.vehicle.get_transform().rotation.yaw
                print(f"! wrong_way triggered: road_id={current_lane_wp.road_id if current_lane_wp else 'N/A'}, "
                      f"lane_id={current_lane_wp.lane_id if current_lane_wp else 'N/A'}, "
                      f"is_junction={in_junction}, "
                      f"lane_yaw={current_lane_wp.transform.rotation.yaw if current_lane_wp else float('nan'):.1f}, "
                      f"vehicle_yaw={vehicle_yaw:.1f}, "
                      f"heading_error_deg={math.degrees(obs[5]):.1f}")

            # 3. Reward function: incentivize progress toward the goal, penalize crashes.
            # Round 12: rebalanced back toward a net-positive per-tick reward
            # for competent driving -- rounds 9-11 stacked enough per-tick
            # penalties (lane offset, heading error, steering smoothness,
            # obstacle) that a good drive's total went negative, at which
            # point discounted RL's optimal policy is to end the episode as
            # fast/cheaply as possible (see PROGRESS.md round 12 diagnosis).
            # No flat per-tick speed reward (there used to be one: `speed *
            # 0.1`) -- that let a policy earn thousands of reward just by
            # driving fast and surviving long, dwarfing terminal penalties
            # (round 8's reward-hacking bug). Progress-toward-goal shaping
            # below is the only per-tick driver of reward; it telescopes to a
            # total bounded by net distance closed, so it can't be inflated
            # by just running out the clock.
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

            reward = -0.01  # Small per-step penalty to encourage efficiency

            # Penalty for idling (not moving AND not accelerating)
            if speed < 1.0 and throttle_action < 0.1:
                reward -= 0.3  # Mild penalty for laziness

            # Potential-based shaping on distance-to-goal: telescopes to a
            # total bounded by net distance closed over the episode.
            # Coefficient raised 0.5 -> 1.0 (round 12) so a full route's
            # worth of progress is a clearly positive, dominant total rather
            # than being swamped by the per-tick penalties below.
            distance_to_goal = obs[3]
            reward += 1.0 * (self.prev_distance_to_goal - distance_to_goal)
            self.prev_distance_to_goal = float(distance_to_goal)

            # Advance waypoint if we're close enough (only if actually moving).
            # Milestone bonus, not the main reward driver.
            if len(self.waypoints) > self.waypoint_index:
                distance_to_waypoint = obs[1]
                if distance_to_waypoint < 2.0 and speed > 2.0:  # Within 2m AND moving at least 2 km/h
                    self.waypoint_index += 1
                    reward += 2.0

            # Reward for heading toward waypoint (only if moving).
            if speed > 1.0:  # Only reward steering when actually moving
                angle_error = abs(obs[2])
                reward += max(0, 0.5 - angle_error / np.pi) * 0.1

            # Continuous penalty for drifting off lane center / misaligning
            # with road heading. Coefficient brought back down 0.25 -> 0.15
            # (round 12) -- round 11's 0.25 was calibrated in isolation and,
            # stacked with every other penalty below, was part of why total
            # reward went negative for normal driving.
            lane_offset = obs[4]
            heading_error = obs[5]
            reward -= 0.15 * abs(lane_offset)
            reward -= 0.1 * abs(heading_error)

            # Penalize crossing a solid lane marking (wrong-lane / off-road
            # edge). Round 12: dropped plain `Broken` (dashed) back out of
            # the trigger set (see _on_lane_invasion) -- dashed lines are
            # what separate most same-direction lanes, and crossing one is
            # *mandatory* when making a legal turn through most junctions,
            # so penalizing it was directly punishing turning.
            if self.lane_invaded_this_step:
                reward -= 2.0

            # Round 12: the tick-to-tick steering-smoothness penalty (rounds
            # 9-11) is removed. It was computed on the *sampled* action, so
            # it penalized the policy's own Gaussian exploration noise, not
            # just genuine erratic driving -- fighting ent_coef for control
            # of the action distribution's variance (see PROGRESS.md round
            # 12 diagnosis). Jitter is now addressed structurally instead,
            # via action_repeat + the steer low-pass filter above.

            # Obstacle-awareness shaping: obs[8] is the distance to the
            # nearest detected obstacle ahead, sentinel-capped at
            # self.obstacle_lookahead when nothing is in range. Penalize
            # closing fast on a near obstacle, reward braking proportionally
            # -- but only while actually moving (speed > 2 km/h), so this
            # can't be farmed by parking near a static prop and holding the
            # brake forever (a round 9-11 exploit structurally identical to
            # the old flat speed-reward hack).
            obstacle_distance = obs[8]
            if obstacle_distance < self.obstacle_lookahead and speed > 2.0:
                danger = (self.obstacle_lookahead - obstacle_distance) / self.obstacle_lookahead  # 0..1
                reward -= danger * speed * 0.02
                reward += danger * brake_action * 0.3

            # 4. Check terminal criteria
            # Track sustained near-zero speed regardless of throttle level.
            # Nothing legitimate about this route requires the car to stop
            # (no traffic lights/stop signs), so sustained near-zero speed
            # means stuck, whatever the throttle is doing.
            if speed < 0.1:
                self.stall_counter += 1
            else:
                self.stall_counter = 0

            # Round 12: terminal penalties cut way down (150 -> 30, uniform
            # across bad outcomes; timeout 75 -> 20). With per-tick reward
            # now net-positive for decent driving, ending the episode early
            # already forfeits all the future positive reward that would
            # otherwise accrue (discounted opportunity cost) -- that's
            # already a strong, sufficient disincentive against dying, so a
            # large terminal penalty stacked on top isn't needed and, worse,
            # previously *inverted* the incentive once per-tick reward was
            # negative (see PROGRESS.md round 12 diagnosis: rounds 9-11's
            # best-ever discovered outcome was stalling immediately).
            if self.crashed:
                reward -= 30.0
                terminated = True
                termination_reason = "crash"
            elif self.off_road:
                # Diagnostic logging: off_road now only fires when the
                # vehicle's center is outside every drivable lane polygon on
                # a *non-junction* segment, so this should mostly be genuine
                # grass/sidewalk exits -- if dist_to_lane is small or road_id
                # looks junction-adjacent, the detection boundary is still
                # wrong somewhere.
                vehicle_loc = self.vehicle.get_location()
                if current_lane_wp is not None:
                    projected_loc = current_lane_wp.transform.location
                    dist_to_lane = math.sqrt((vehicle_loc.x - projected_loc.x)**2 +
                                             (vehicle_loc.y - projected_loc.y)**2)
                    road_id = current_lane_wp.road_id
                    lane_id = current_lane_wp.lane_id
                else:
                    dist_to_lane = float("nan")
                    road_id = lane_id = "N/A"
                print(f"! off_road triggered: vehicle at ({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f}, "
                      f"{vehicle_loc.z:.1f}), not on any drivable lane (project_to_road=False -> None), "
                      f"nearest drivable lane: road_id={road_id}, lane_id={lane_id}, "
                      f"is_junction={in_junction}, dist_to_lane={dist_to_lane:.1f}m, "
                      f"lateral_offset={obs[4]:.2f}m, heading_error_deg={math.degrees(obs[5]):.1f}, "
                      f"speed={speed:.1f}km/h, episode_steps={self.episode_steps}")
                reward -= 30.0
                terminated = True
                termination_reason = "off_road"
            elif self.wrong_way:
                reward -= 30.0
                terminated = True
                termination_reason = "wrong_way"
            elif self.stall_counter >= 20:  # ~1 second of sustained stalling
                reward -= 30.0
                terminated = True
                termination_reason = "stall"
            elif self.waypoint_index >= len(self.waypoints):
                # Reached the end of the route.
                reward += 500.0
                terminated = True
                termination_reason = "success"

            # 5. Episode truncation (time limit)
            if self.episode_steps >= self.max_episode_steps:
                truncated = True
                reward -= 20.0
                termination_reason = "timeout"

            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, {"termination_reason": termination_reason}

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

        obstacle_distance = self._distance_to_obstacle_ahead(vehicle_loc, vehicle_yaw)

        return np.array([speed, distance_to_waypoint, angle_to_waypoint, distance_to_goal,
                          lane_offset, heading_error, self.prev_steer, self.prev_throttle,
                          obstacle_distance],
                         dtype=np.float32)

    def _distance_to_obstacle_ahead(self, vehicle_loc, vehicle_yaw_deg, half_angle_deg=25.0):
        """Distance to the nearest obstacle roughly in front of the vehicle,
        within self.obstacle_lookahead meters -- checks both static level
        geometry (e.g. parked-car props baked into the map, which never show
        up in world.get_actors()) and any dynamic vehicle actors (e.g. NPC
        traffic, if ever added). Returns self.obstacle_lookahead (the "clear"
        sentinel) if nothing is in range."""
        max_distance = self.obstacle_lookahead
        yaw_rad = math.radians(vehicle_yaw_deg)
        forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)

        nearest = max_distance

        def consider(obstacle_loc):
            nonlocal nearest
            dx = obstacle_loc.x - vehicle_loc.x
            dy = obstacle_loc.y - vehicle_loc.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 1e-3 or dist >= nearest:
                return
            angle = math.degrees(math.acos(np.clip((dx * forward_x + dy * forward_y) / dist, -1.0, 1.0)))
            if angle <= half_angle_deg:
                nearest = dist

        for bbox in self.static_vehicle_bboxes:
            consider(bbox.location)

        for actor in self.world.get_actors().filter("vehicle.*"):
            if self.vehicle is not None and actor.id == self.vehicle.id:
                continue
            consider(actor.get_location())

        return nearest

    def _on_collision(self, event):
        self.crashed = True

    def _on_lane_invasion(self, event):
        # Round 12: dropped plain `Broken` (dashed) back out -- see the
        # reward-function comment in step() for why.
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
        # Always restore rendering on exit, even if this instance disabled
        # it -- it's a server-wide setting, so leaving it off would silently
        # break camera output for whatever connects next (e.g. drive.py).
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)