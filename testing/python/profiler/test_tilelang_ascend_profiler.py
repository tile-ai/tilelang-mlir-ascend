import contextlib
import sys
import types

import pytest

import tilelang.profiler.npu as npu_profiler
import tilelang.opentile.tile_obj as tile_obj_module
from tilelang.jit.adapter.tile import TileKernelAdapter
from tilelang.opentile.tile_obj import TileObjectKernel


def _report(case, **details):
    print(f"\n[NPU Profiler UT] {case}")
    for name, value in details.items():
        print(f"  - {name}: {value}")


def _capture_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        npu_profiler.logger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    return warnings


def _environment(tmp_path, profile="1", strict="0"):
    return types.SimpleNamespace(
        TILELANG_NPU_PROFILE=profile,
        TILELANG_NPU_PROFILE_DIR=str(tmp_path),
        TILELANG_NPU_PROFILE_SKIP_FIRST="0",
        TILELANG_NPU_PROFILE_WARMUP="0",
        TILELANG_NPU_PROFILE_ACTIVE="1",
        TILELANG_NPU_PROFILE_ANALYSE="1",
        TILELANG_NPU_PROFILE_STRICT=strict,
    )


@pytest.fixture
def fake_torch_npu(monkeypatch):
    events = []

    class FakeProfileInstance:
        def start(self):
            events.append("start")

        def step(self):
            events.append("step")

        def stop(self):
            events.append("stop")

    class FakeProfiler:
        class ProfilerActivity:
            CPU = "CPU"
            NPU = "NPU"

        class ProfilerLevel:
            Level1 = "Level1"

        class AiCMetrics:
            PipeUtilization = "PipeUtilization"

        @staticmethod
        def _ExperimentalConfig(**kwargs):
            events.append(("experimental_config", kwargs))
            return kwargs

        @staticmethod
        def schedule(**kwargs):
            events.append(("schedule", kwargs))
            return kwargs

        @staticmethod
        def tensorboard_trace_handler(path, analyse_flag):
            events.append(("handler", path, analyse_flag))
            return object()

        @staticmethod
        def profile(**kwargs):
            events.append(("profile", kwargs))
            return FakeProfileInstance()

    module = types.ModuleType("torch_npu")
    module.profiler = FakeProfiler
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return events


def _install_fake_session(
    monkeypatch,
    *,
    start_error=None,
    step_error=None,
    stop_error=None,
    finish_on_step=False,
):
    events = []

    class FakeSession:
        def __init__(self, config):
            self.finished = False

        def start(self):
            events.append("start")
            if start_error is not None:
                raise start_error

        def step(self):
            events.append("step")
            if step_error is not None:
                raise step_error
            self.finished = finish_on_step

        def stop(self):
            events.append("stop")
            if stop_error is not None:
                raise stop_error

    monkeypatch.setattr(npu_profiler, "NPUProfilerSession", FakeSession)
    return events


def _make_scalar_adapter():
    adapter = object.__new__(TileKernelAdapter)
    adapter.input_idx = [0]
    adapter.params = [object()]
    adapter.tensor_params = set()
    adapter.scalar_kinds = {0: "int32"}
    adapter.launch_spec = types.SimpleNamespace(kernel_name="fake_operator")
    return adapter


def test_npu_profile_config_accepts_only_zero_or_one(tmp_path):
    disabled = npu_profiler.NPUProfileConfig.from_env(
        _environment(tmp_path, profile="0")
    )
    enabled = npu_profiler.NPUProfileConfig.from_env(
        _environment(tmp_path, profile="1")
    )

    assert disabled.enabled is False
    assert enabled.enabled is True

    with pytest.raises(ValueError, match="expected 0 or 1"):
        npu_profiler.NPUProfileConfig.from_env(
            _environment(tmp_path, profile="on")
        )

    _report(
        "configuration",
        disabled=disabled.enabled,
        enabled=enabled.enabled,
        invalid_value="on -> ValueError",
    )


def test_npu_profiler_session_lifecycle(tmp_path, fake_torch_npu):
    config = npu_profiler.NPUProfileConfig(
        enabled=True,
        output_dir=tmp_path,
        skip_first=1,
        warmup=1,
        active=2,
    )
    session = npu_profiler.NPUProfilerSession(config)

    session.start()
    session.start()
    for _ in range(3):
        session.step()
        assert session.finished is False
    session.step()
    session.step()
    session.stop()

    lifecycle = [event for event in fake_torch_npu if isinstance(event, str)]
    assert lifecycle == ["start", "step", "step", "step", "step", "stop"]
    schedule = next(event[1] for event in fake_torch_npu if event[0] == "schedule")
    assert schedule == {
        "wait": 0,
        "warmup": 1,
        "active": 2,
        "repeat": 1,
        "skip_first": 1,
    }

    experimental_config = next(
        event[1]
        for event in fake_torch_npu
        if event[0] == "experimental_config"
    )
    assert experimental_config == {
        "profiler_level": "Level1",
        "aic_metrics": "PipeUtilization",
    }

    profile_kwargs = next(
        event[1]
        for event in fake_torch_npu
        if event[0] == "profile"
    )
    assert profile_kwargs["experimental_config"] is experimental_config

    _report(
        "session lifecycle",
        total_steps=session.total_steps,
        schedule=schedule,
        experimental_config=experimental_config,
        events=lifecycle,
    )


def test_controller_finishes_one_session_and_does_not_restart(monkeypatch, tmp_path):
    events = _install_fake_session(monkeypatch, finish_on_step=True)
    controller = npu_profiler.NPUProfilerController()
    config = npu_profiler.NPUProfileConfig(enabled=True, output_dir=tmp_path)
    calls = []

    assert controller.run(config, lambda: calls.append(1) or "first") == "first"
    assert controller.run(config, lambda: calls.append(2) or "second") == "second"

    assert calls == [1, 2]
    assert events == ["start", "step"]
    _report(
        "completed session is not restarted",
        callback_calls=len(calls),
        profiler_events=events,
    )


@pytest.mark.parametrize("strict", [False, True])
def test_controller_start_failure(monkeypatch, tmp_path, strict):
    warnings = _capture_warnings(monkeypatch)
    events = _install_fake_session(
        monkeypatch,
        start_error=RuntimeError("start failed"),
    )
    controller = npu_profiler.NPUProfilerController()
    config = npu_profiler.NPUProfileConfig(
        enabled=True,
        output_dir=tmp_path,
        strict=strict,
    )
    calls = []

    if strict:
        with pytest.raises(RuntimeError, match="start failed"):
            controller.run(config, lambda: calls.append(1))
        assert calls == []
    else:
        assert controller.run(config, lambda: calls.append(1) or "ok") == "ok"
        assert calls == [1]

    assert events == ["start"]
    assert bool(warnings) is (not strict)
    _report(
        "controller start failure",
        strict=strict,
        callback_calls=len(calls),
        profiler_events=events,
        warnings=warnings,
    )


@pytest.mark.parametrize("strict", [False, True])
def test_controller_step_failure_does_not_repeat_callback(monkeypatch, tmp_path, strict):
    warnings = _capture_warnings(monkeypatch)
    events = _install_fake_session(
        monkeypatch,
        step_error=RuntimeError("step failed"),
    )
    controller = npu_profiler.NPUProfilerController()
    config = npu_profiler.NPUProfileConfig(
        enabled=True,
        output_dir=tmp_path,
        strict=strict,
    )
    calls = []

    if strict:
        with pytest.raises(RuntimeError, match="step failed"):
            controller.run(config, lambda: calls.append(1) or "ok")
    else:
        assert controller.run(config, lambda: calls.append(1) or "ok") == "ok"

    assert calls == [1]
    assert events == ["start", "step", "stop"]
    assert bool(warnings) is (not strict)
    _report(
        "controller step failure",
        strict=strict,
        callback_calls=len(calls),
        profiler_events=events,
        warnings=warnings,
    )


def test_launch_exception_wins_over_stop_exception(monkeypatch, tmp_path):
    warnings = _capture_warnings(monkeypatch)
    events = _install_fake_session(
        monkeypatch,
        stop_error=RuntimeError("stop failed"),
    )
    controller = npu_profiler.NPUProfilerController()
    config = npu_profiler.NPUProfileConfig(enabled=True, output_dir=tmp_path)
    calls = []

    def launch():
        calls.append(1)
        raise ValueError("launch failed")

    with pytest.raises(ValueError, match="launch failed"):
        controller.run(config, launch)

    assert calls == [1]
    assert events == ["start", "stop"]
    assert warnings == ["Failed to close NPU profiler"]
    _report(
        "launch exception has priority",
        raised="ValueError: launch failed",
        suppressed_cleanup_error="RuntimeError: stop failed",
        callback_calls=len(calls),
        profiler_events=events,
        warnings=warnings,
    )


def test_adapter_disabled_bypasses_controller(monkeypatch):
    monkeypatch.setenv("TILELANG_NPU_PROFILE", "0")

    class ForbiddenController:
        def run(self, config, callback):
            pytest.fail("disabled path must not enter the profiler controller")

    monkeypatch.setattr(
        npu_profiler,
        "_npu_profiler_controller",
        ForbiddenController(),
    )
    adapter = _make_scalar_adapter()
    calls = []
    adapter._invoke_validated = lambda logical, device: calls.append(logical) or "ok"

    assert adapter._convert_torch_func()(7) == "ok"
    assert calls == [[7]]
    _report(
        "disabled adapter path",
        controller_calls=0,
        invoke_calls=len(calls),
    )


def test_adapter_enabled_enters_controller_and_invokes_once(monkeypatch, tmp_path):
    monkeypatch.setenv("TILELANG_NPU_PROFILE", "1")
    monkeypatch.setenv("TILELANG_NPU_PROFILE_DIR", str(tmp_path))

    class FakeController:
        def __init__(self):
            self.calls = 0

        def run(self, config, callback):
            self.calls += 1
            assert config.enabled is True
            return callback()

    controller = FakeController()
    monkeypatch.setattr(
        npu_profiler,
        "_npu_profiler_controller",
        controller,
    )
    adapter = _make_scalar_adapter()
    calls = []
    adapter._invoke_validated = lambda logical, device: calls.append(logical) or "ok"

    assert adapter._convert_torch_func()(7) == "ok"
    assert controller.calls == 1
    assert calls == [[7]]
    _report(
        "enabled adapter path",
        controller_calls=controller.calls,
        invoke_calls=len(calls),
    )


@pytest.mark.parametrize(
    ("profile_value", "expected_annotations"),
    [("0", []), ("1", ["tilelang.launch::fake_kernel"])],
)
def test_tile_obj_launches_once(monkeypatch, profile_value, expected_annotations):
    monkeypatch.setenv("TILELANG_NPU_PROFILE", profile_value)
    monkeypatch.setattr(tile_obj_module, "_current_npu_stream", lambda: 0)
    annotations = []

    @contextlib.contextmanager
    def fake_annotation(name, **kwargs):
        annotations.append(name)
        yield

    monkeypatch.setattr(npu_profiler, "npu_annotation", fake_annotation)

    kernel = object.__new__(TileObjectKernel)
    kernel.kernel_name = "fake_kernel"
    kernel._kernel_token = 1
    kernel.block_count = 1
    kernel.dynamic_ubuf_bytes = 0
    kernel._compute_dynamic_ubuf_size = lambda: 0
    kernel._encode_args = lambda args: args
    launches = []
    kernel._launch = lambda *args: launches.append(args)

    kernel(7)

    assert len(launches) == 1
    assert annotations == expected_annotations
    _report(
        "TileObjectKernel launch",
        profile_value=profile_value,
        launch_count=len(launches),
        annotations=annotations,
    )
