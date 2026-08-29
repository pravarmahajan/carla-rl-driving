from pathlib import Path
import tempfile
import unittest

from carla_rl.config import ServerCatalog
from carla_rl.server_slots import NoServerSlotAvailable, acquire_server


REPOSITORY = Path(__file__).resolve().parents[1]


class ServerLeaseTests(unittest.TestCase):
    def test_auto_lease_skips_a_slot_held_by_this_process(self):
        catalog = ServerCatalog.load(REPOSITORY / "configs" / "servers.json")
        with tempfile.TemporaryDirectory() as directory:
            lock_dir = Path(directory)
            with acquire_server(catalog, "sim-0", lock_dir) as first:
                self.assertEqual(first.name, "sim-0")
                with acquire_server(catalog, "auto", lock_dir) as second:
                    self.assertEqual(second.name, "sim-1")

    def test_explicit_lease_fails_when_slot_is_held(self):
        catalog = ServerCatalog.load(REPOSITORY / "configs" / "servers.json")
        with tempfile.TemporaryDirectory() as directory:
            lock_dir = Path(directory)
            with acquire_server(catalog, "sim-0", lock_dir):
                with self.assertRaises(NoServerSlotAvailable):
                    with acquire_server(catalog, "sim-0", lock_dir):
                        pass


if __name__ == "__main__":
    unittest.main()

