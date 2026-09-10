import unittest

from carla_rl.scenarios import ScenarioConfigurationError, ScenarioSpec


class ScenarioSpecTests(unittest.TestCase):
    def test_parses_a_replayable_static_actor_scenario(self):
        scenario = ScenarioSpec.from_mapping({
            "id": "lead-vehicle-v0",
            "ego_start": {"x": 1, "y": 2, "z": 0.5, "yaw": 90},
            "goal": {"x": 100, "y": 2, "z": 0.5},
            "actors": [{
                "name": "lead_vehicle",
                "blueprint": "vehicle.tesla.model3",
                "transform": {"x": 15, "y": 2, "z": 0.5, "yaw": 90},
                "attributes": {"role_name": "scenario_lead"},
            }],
        })
        self.assertEqual(scenario.scenario_id, "lead-vehicle-v0")
        self.assertEqual(scenario.ego_start.yaw, 90.0)
        self.assertEqual(scenario.goal.x, 100.0)
        self.assertEqual(scenario.actors[0].attributes["role_name"], "scenario_lead")

    def test_rejects_duplicate_actor_names(self):
        with self.assertRaisesRegex(ScenarioConfigurationError, "unique"):
            ScenarioSpec.from_mapping({
                "id": "duplicate-actors",
                "actors": [
                    {"name": "a", "blueprint": "vehicle.a", "transform": {"x": 0, "y": 0, "z": 0}},
                    {"name": "a", "blueprint": "vehicle.b", "transform": {"x": 1, "y": 0, "z": 0}},
                ],
            })

    def test_rejects_implicit_actor_behavior(self):
        with self.assertRaisesRegex(ScenarioConfigurationError, "unsupported"):
            ScenarioSpec.from_mapping({
                "id": "not-yet",
                "actors": [{
                    "name": "lead", "blueprint": "vehicle.a",
                    "transform": {"x": 0, "y": 0, "z": 0}, "autopilot": True,
                }],
            })


if __name__ == "__main__":
    unittest.main()
