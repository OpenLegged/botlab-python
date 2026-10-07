"""Dependency probing for the botlab local runtime (deps-free importable).

Tier-A tests run without extras installed: the pure decision functions are
tested directly; the import probe is tested against the real environment.
"""

import unittest

from botlab._runtime import _deps  # noqa: E402


class MissingDependencyMessageTest(unittest.TestCase):
    def test_reports_install_command_for_each_missing_package(self):
        report = _deps.describe_missing(["mujoco", "onnxruntime", "websockets"])
        self.assertIn('pip install "botlab-python[runtime]"', report)
        self.assertIn("mujoco", report)
        self.assertIn("onnxruntime", report)
        self.assertIn("websockets", report)

    def test_no_missing_returns_none(self):
        self.assertIsNone(_deps.describe_missing([]))

    def test_message_names_the_failed_capability(self):
        report = _deps.describe_missing(["mujoco"])
        self.assertIn("local simulation", report)


class ExecutionProviderSelectionTest(unittest.TestCase):
    def test_cuda_preferred_when_available(self):
        providers, warning = _deps.select_execution_providers(
            ["CPUExecutionProvider", "CUDAExecutionProvider"]
        )
        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertIsNone(warning)

    def test_cpu_only_has_no_warning(self):
        providers, warning = _deps.select_execution_providers(["CPUExecutionProvider"])
        self.assertEqual(providers, ["CPUExecutionProvider"])
        self.assertIsNone(warning)

    def test_missing_cuda_when_gpu_requested_warns_and_falls_back(self):
        providers, warning = _deps.select_execution_providers(
            ["CPUExecutionProvider"],
            requested="cuda",
        )
        self.assertEqual(providers, ["CPUExecutionProvider"])
        self.assertIsNotNone(warning)
        self.assertIn("CUDAExecutionProvider", warning)
        self.assertIn("onnxruntime-gpu", warning)


class ImportProbeTest(unittest.TestCase):
    def test_probe_returns_available_and_missing_lists(self):
        probe = _deps.probe_runtime_deps()
        # Tier A is dependency-free: every runtime requirement must be reported
        # in exactly one list, regardless of what the host happens to install.
        self.assertFalse(set(probe.available) & set(probe.missing))
        self.assertEqual(
            sorted(probe.available + probe.missing),
            sorted(_deps.RUNTIME_REQUIREMENTS),
        )

    def test_runtime_error_when_missing_names_install_command(self):
        with self.assertRaises(_deps.RuntimeDependenciesError) as ctx:
            _deps.require_runtime_deps(
                available=["numpy"],
                missing=["mujoco", "onnxruntime"],
            )
        self.assertIn('pip install "botlab-python[runtime]"', str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
