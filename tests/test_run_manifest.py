from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from carla_rl.config import ExperimentConfig, ServerCatalog
from carla_rl.run_manifest import create_run


REPOSITORY = Path(__file__).resolve().parents[1]


class RunManifestTests(unittest.TestCase):
    def test_create_run_records_server_config_and_git_state(self):
        experiment = ExperimentConfig.load(
            REPOSITORY / "configs" / "experiments" / "baseline_state_v1.json"
        )
        catalog = ServerCatalog.load(REPOSITORY / "configs" / "servers.json")
        fixed_time = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = create_run(
                experiment,
                catalog,
                "sim-2",
                Path(directory),
                repository=REPOSITORY,
                run_name="Manifest Test",
                command="evaluate",
                runtime_overrides={"episodes": 3, "model_path": "models/candidate"},
                now=fixed_time,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["server_catalog"]["selected"], "sim-2")
            self.assertEqual(manifest["command"], "evaluate")
            self.assertEqual(resolved["simulator"]["rpc_port"], 2020)
            self.assertEqual(resolved["runtime"]["episodes"], 3)
            self.assertTrue((run_dir / "git.patch").exists())
            self.assertIn("carla_python_agents", manifest["external_sources"])
            self.assertIn("src/carla_rl/run_manifest.py", (run_dir / "git.patch").read_text(encoding="utf-8"))
            self.assertNotEqual(
                manifest["experiment"]["fingerprint"],
                manifest["experiment"]["resolved_fingerprint"],
            )

            with self.assertRaises(FileExistsError):
                create_run(
                    experiment,
                    catalog,
                    "sim-2",
                    Path(directory),
                    repository=REPOSITORY,
                    run_name="Manifest Test",
                    now=fixed_time,
                )


if __name__ == "__main__":
    unittest.main()
