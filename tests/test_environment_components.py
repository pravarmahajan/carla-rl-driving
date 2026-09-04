import math
import unittest

import numpy as np

from carla_rl.environment_components import (
    ObservationInputs,
    Round12Reward,
    RewardInputs,
    Round12Termination,
    TerminationInputs,
    heading_error_rad,
)


class StateV1CharacterizationTests(unittest.TestCase):
    """Numerical guardrails for the frozen Round16b environment contracts."""

    def test_observation_feature_order_is_frozen(self):
        obs = ObservationInputs(12, 3, -0.2, 80, 0.5, 0.1, -0.3, 0.7, 30).as_state_v1()
        self.assertEqual(obs.dtype.name, "float32")
        np.testing.assert_allclose(obs, [12.0, 3.0, -0.2, 80.0, 0.5, 0.1, -0.3, 0.7, 30.0])

    def test_reward_matches_round12_reference_transition(self):
        reward = Round12Reward().evaluate(
            RewardInputs(
                speed_kmh=36.0, throttle=0.6, brake=0.0,
                previous_distance_to_goal_m=100.0, distance_to_goal_m=97.5,
                distance_to_waypoint_m=1.0, route_heading_error_rad=0.0,
                lane_offset_m=0.2, lane_heading_error_rad=0.1,
                obstacle_distance_m=30.0, lane_invaded=False, reached_waypoint=True,
            )
        )
        self.assertAlmostEqual(reward, 4.5, places=7)

    def test_termination_precedence_and_timeout_are_frozen(self):
        crash = Round12Termination().evaluate(TerminationInputs(True, True, True, 30, True, 1499, 1500))
        self.assertEqual((crash.terminated, crash.truncated, crash.reason, crash.reward_adjustment), (True, False, "crash", -30.0))
        timeout = Round12Termination().evaluate(TerminationInputs(False, False, False, 0, False, 1500, 1500))
        self.assertEqual((timeout.terminated, timeout.truncated, timeout.reason, timeout.reward_adjustment), (False, True, "timeout", -20.0))
        # The legacy code runs its time-limit check after a terminal check.
        boundary = Round12Termination().evaluate(TerminationInputs(True, False, False, 0, False, 1500, 1500))
        self.assertEqual((boundary.terminated, boundary.truncated, boundary.reason, boundary.reward_adjustment), (True, True, "timeout", -50.0))

    def test_heading_error_wraps_at_map_boundary(self):
        self.assertAlmostEqual(heading_error_rad(-179, 179), math.radians(2), places=7)


if __name__ == "__main__":
    unittest.main()
