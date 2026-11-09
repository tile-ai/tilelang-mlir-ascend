"""CPU protocol checks with a fake torch_npu; no hardware collection claims."""

import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "assets/skills/harness-develop/scripts"))
import torch_npu_diagnostics as diagnostics


class DiagnosticToolsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []
        self.configs = []
        self.identity = {key: f"synthetic-{key}" for key in ("candidate", "case", "build", "environment")}
        self.fake = self.profiler()

    def profiler(self):
        def handler(directory, **kwargs):
            self.assertEqual(kwargs, {"analyse_flag": True, "async_mode": False})
            return Path(directory)

        @contextlib.contextmanager
        def profile(**kwargs):
            self.events.append("profile-start")
            self.configs.append(kwargs)
            yield SimpleNamespace(step=lambda: self.events.append("step"))
            directory = kwargs["on_trace_ready"]
            (directory / "trace_view.json").write_text('{"traceEvents": []}')
            self.events.append("profile-stop")

        return SimpleNamespace(
            __version__="synthetic", profiler=SimpleNamespace(
                AiCMetrics=SimpleNamespace(PipeUtilization=1, Memory=2),
                supported_ai_core_metrics=lambda: {1, 2},
                ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
                ProfilerLevel=SimpleNamespace(Level1="level1"),
                ExportType=SimpleNamespace(Text="text"),
                _ExperimentalConfig=lambda **kwargs: kwargs,
                schedule=lambda **kwargs: kwargs,
                tensorboard_trace_handler=handler, profile=profile),
            npu=SimpleNamespace(mstx=SimpleNamespace(
                range_start=lambda name: self.events.append(("start", name)) or 7,
                range_end=lambda ident: self.events.append(("end", ident)))))

    def callbacks(self):
        return {"prepare": lambda: self.events.append("prepare"),
                "synchronize": lambda: self.events.append("sync"),
                "check_reservation": lambda: self.events.append("reservation")}

    def capture(self, **kwargs):
        options = dict(output=self.root / "native", identity=self.identity,
                       metrics=("PipeUtilization", "Memory"), warmup=1, **self.callbacks())
        options.update(kwargs)
        with patch.dict(sys.modules, {"torch_npu": self.fake}):
            return diagnostics.capture_diagnostics(lambda: self.events.append("call"), **options)

    def test_native_passes_reset_and_keep_raw_evidence_pending_review(self):
        result = self.capture()
        self.assertEqual(result["status"], "pending_review")
        self.assertEqual(result["identity"], self.identity)
        self.assertEqual([row["metric"] for row in result["passes"]], ["PipeUtilization", "Memory"])
        self.assertEqual(self.events.count("prepare"), 3)
        self.assertEqual(self.events.count("call"), 3)
        for index, event in enumerate(self.events):
            if event == "profile-start":
                self.assertEqual(self.events[index - 2:index], ["prepare", "sync"])
                self.assertEqual(self.events[index:index + 5],
                                 ["profile-start", "call", "sync", "step", "profile-stop"])
        self.assertEqual([x["experimental_config"]["aic_metrics"] for x in self.configs], [1, 2])
        for config in self.configs:
            self.assertFalse(config["experimental_config"]["data_simplification"])
            self.assertEqual(config["schedule"]["active"], 1)
        self.assertTrue(result["passes"][0]["files"][0]["sha256"])
        # A file existing does not establish real kernel coverage: no automatic ready status.
        with self.assertRaises(FileExistsError):
            self.capture()

    def test_msprof_metrics_are_not_silently_sent_to_native_profiler(self):
        for metrics in (("PipeTimeline",), ("Occupancy",), ("Roofline",), ("Default",),
                        "PipeUtilization,Memory", ("Memory", "Memory")):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                self.capture(metrics=metrics)
        self.assertEqual(self.events, [])

    def test_unsupported_native_metric_preserves_failure_without_calls(self):
        self.fake.profiler.supported_ai_core_metrics = lambda: {1}
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            self.capture()
        report = json.loads((self.root / "native/collection.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["passes"], [])
        self.assertNotIn("call", self.events)

    def test_empty_export_is_failure_and_previous_pass_is_retained(self):
        original = self.fake.profiler.profile

        @contextlib.contextmanager
        def profile(**kwargs):
            if kwargs["experimental_config"]["aic_metrics"] == 2:
                yield SimpleNamespace(step=lambda: None)
            else:
                with original(**kwargs) as prof:
                    yield prof

        self.fake.profiler.profile = profile
        with self.assertRaisesRegex(RuntimeError, "no diagnostic artifacts"):
            self.capture()
        report = json.loads((self.root / "native/collection.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["passes"]), 1)
        self.assertTrue((self.root / "native/PipeUtilization/trace_view.json").exists())

    def test_mstx_range_excludes_preparation_and_always_ends(self):
        def call():
            self.events.append("call")
            if self.events.count("call") == 2:
                raise RuntimeError("candidate failure")
        with patch.dict(sys.modules, {"torch_npu": self.fake}):
            with self.assertRaisesRegex(RuntimeError, "candidate failure"):
                diagnostics.marked_call(call, range_name="harness_target", warmup=1, **self.callbacks())
        start = self.events.index(("start", "harness_target"))
        self.assertEqual(self.events[start - 2:start], ["prepare", "sync"])
        self.assertEqual(self.events[start + 1:], ["call", ("end", 7)])

    def test_invalid_mstx_id_prevents_target_call(self):
        for value in (None, 0, -1, True):
            self.fake.npu.mstx.range_start = lambda name: value
            self.events.clear()
            with patch.dict(sys.modules, {"torch_npu": self.fake}):
                with self.assertRaisesRegex(RuntimeError, "did not start"):
                    diagnostics.marked_call(lambda: self.events.append("call"),
                                            range_name="target", warmup=0, **self.callbacks())
            self.assertNotIn("call", self.events)

    def test_msprof_argv_preserves_application_arguments_and_restricts_matching(self):
        application = ["python3", "worker with spaces.py", "--case", "case;literal"]
        argv = diagnostics.msprof_command(application, kernel_name="real_kernel_42",
                                          output=self.root / "op", range_name="target")
        self.assertEqual(argv[:2], ["msprof", "op"])
        self.assertEqual(argv[-4:], application)
        self.assertIn("--aic-metrics=Default,PipeTimeline,Occupancy,Roofline,Memory", argv)
        self.assertIn("--mstx-include=target", argv)
        self.assertIn("--launch-count=1", argv)
        for key, value in (("kernel_name", "*"), ("range_name", "a|b"),
                           ("replay_mode", "range"), ("launch_count", 0),
                           ("launch_skip_before_match", -1), ("launch_skip_before_match", 1001)):
            options = dict(kernel_name="target", range_name="target", output=self.root)
            options[key] = value
            with self.assertRaises(ValueError):
                diagnostics.msprof_command(application, **options)

    def test_msprof_without_mstx_uses_explicit_launch_selection(self):
        argv = diagnostics.msprof_command(["python3", "worker.py"], kernel_name="kernel",
                                          output=self.root, launch_skip_before_match=12)
        self.assertIn("--launch-skip-before-match=12", argv)
        self.assertFalse(any(arg.startswith("--mstx") for arg in argv))

    def test_artifact_inventory_is_relative_hashed_and_rejects_links(self):
        (self.root / "trace.json").write_text("synthetic")
        records = diagnostics.artifact_index(self.root)
        self.assertEqual(records[0]["path"], "trace.json")
        self.assertEqual(records[0]["bytes"], 9)
        self.assertEqual(len(records[0]["sha256"]), 64)
        (self.root / "alias").symlink_to(self.root / "trace.json")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            diagnostics.artifact_index(self.root)


if __name__ == "__main__":
    unittest.main()
