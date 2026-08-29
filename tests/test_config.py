import json
from pathlib import Path
import tempfile
import unittest

from carla_rl.config import ConfigurationError, ExperimentConfig, ServerCatalog


REPOSITORY = Path(__file__).resolve().parents[1]


class ServerCatalogTests(unittest.TestCase):
    def test_repository_catalog_has_three_disjoint_slots(self):
        catalog = ServerCatalog.load(REPOSITORY / "configs" / "servers.json")
        self.assertEqual([server.name for server in catalog.servers], ["sim-0", "sim-1", "sim-2"])
        self.assertEqual(catalog.get("sim-2").rpc_port, 2020)

    def test_duplicate_ports_are_rejected(self):
        data = {
            "schema_version": 1,
            "servers": [
                {
                    "name": "a",
                    "host": "127.0.0.1",
                    "rpc_port": 2000,
                    "streaming_port": 2001,
                    "secondary_port": 2002,
                    "traffic_manager_port": 8000,
                    "default_role": "training",
                },
                {
                    "name": "b",
                    "host": "127.0.0.1",
                    "rpc_port": 2000,
                    "streaming_port": 2011,
                    "secondary_port": 2012,
                    "traffic_manager_port": 8010,
                    "default_role": "training",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "servers.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                ServerCatalog.load(path)


class ExperimentConfigTests(unittest.TestCase):
    def test_baseline_schema_records_feature_order(self):
        experiment = ExperimentConfig.load(
            REPOSITORY / "configs" / "experiments" / "baseline_state_v1.json"
        )
        features = experiment.data["environment"]["observation_features"]
        self.assertEqual(len(features), 9)
        self.assertEqual(features[0], "speed_kmh")
        self.assertEqual(features[-1], "obstacle_distance_m")
        self.assertEqual(len(experiment.fingerprint), 64)


if __name__ == "__main__":
    unittest.main()

