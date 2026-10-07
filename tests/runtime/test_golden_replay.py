"""Golden replay tests: Python processors vs TS-exported traces (tier B)."""

import os
import unittest

from botlab._runtime import golden_check  # noqa: E402
from botlab._runtime._processors import build_registry  # noqa: E402

# Types with exported golden cases whose Python port has not landed yet —
# empty since every plan node type now has a port and replays in tolerance.
# A new TS processor type without a fixture+port fails the ratchets above.
EXPECTED_UNPORTED: set[str] = set()


class GoldenReplayTest(unittest.TestCase):
    def test_every_exported_case_replays_within_tolerance(self):
        golden_path = golden_check.default_golden_path()
        self.assertTrue(
            os.path.isfile(golden_path),
            f"missing {golden_path}; run UPDATE_GOLDEN=1 vitest to export",
        )
        failures, total = golden_check.replay(golden_path)
        blocking = [
            failure for failure in failures if failure["type"] not in EXPECTED_UNPORTED
        ]
        self.assertEqual(
            blocking,
            [],
            f"{len(blocking)}/{total} golden cases diverge: "
            + "; ".join(f"{f['type']}: {f['reason']}" for f in blocking[:5]),
        )
        self.assertGreater(total, 0)

    def test_unported_allowlist_only_shrinks(self):
        registry = build_registry()
        still_missing = [
            type_name for type_name in EXPECTED_UNPORTED if type_name not in registry
        ]
        self.assertEqual(
            sorted(EXPECTED_UNPORTED),
            sorted(still_missing),
            "a type in EXPECTED_UNPORTED now has a Python port; remove it from "
            "the allowlist",
        )


if __name__ == "__main__":
    unittest.main()
