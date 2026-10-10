"""Configuration and annotations for optional NPU profiling."""

from __future__ import annotations

import logging
import sys
import atexit
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "NPUProfileConfig",
    "NPUProfilerSession",
    "NPUProfilerController",
    "npu_annotation",
]


def _env_bool(value: str) -> bool:
    normalized = str(value).strip()

    if normalized == "1":
        return True
    if normalized == "0":
        return False

    raise ValueError(f"expected 0 or 1, got {value!r}")


@dataclass(frozen=True)
class NPUProfileConfig:
    enabled: bool = False
    output_dir: str | Path = "./tilelang_profile"

    skip_first: int = 0
    warmup: int = 0
    active: int = 1

    analyse: bool = True
    strict: bool = False

    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir).expanduser(),
        )

    @classmethod
    def from_env(cls, environment=None) -> NPUProfileConfig:
        if environment is None:
            from tilelang.env import env as environment

        return cls(
            enabled=_env_bool(environment.TILELANG_NPU_PROFILE),
            output_dir=environment.TILELANG_NPU_PROFILE_DIR,
            skip_first=int(environment.TILELANG_NPU_PROFILE_SKIP_FIRST),
            warmup=int(environment.TILELANG_NPU_PROFILE_WARMUP),
            active=int(environment.TILELANG_NPU_PROFILE_ACTIVE),
            analyse=_env_bool(environment.TILELANG_NPU_PROFILE_ANALYSE),
            strict=_env_bool(environment.TILELANG_NPU_PROFILE_STRICT),
        )


class NPUProfilerSession:
    """Own one automatically managed TorchNPU profiler session."""

    def __init__(self, config: NPUProfileConfig):
        self.config = config
        self._profiler = None
        self._step_count = 0
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def total_steps(self) -> int:
        return (
            self.config.skip_first
            + self.config.warmup
            + self.config.active
        )

    def start(self) -> None:
        if self._profiler is not None or self._finished:
            return

        # Keep TorchNPU profiler out of the disabled launch path.
        from torch_npu import profiler

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        schedule = profiler.schedule(
            wait=0,
            warmup=self.config.warmup,
            active=self.config.active,
            repeat=1,
            skip_first=self.config.skip_first,
        )

        trace_handler = profiler.tensorboard_trace_handler(
            str(self.config.output_dir),
            analyse_flag=self.config.analyse,
        )

        experimental_config = profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
        )

        profiler_instance = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.NPU,
            ],
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            experimental_config=experimental_config,
        )

        profiler_instance.start()
        self._profiler = profiler_instance

    def step(self) -> None:
        if self._profiler is None or self._finished:
            return

        self._profiler.step()
        self._step_count += 1

        if self._step_count >= self.total_steps:
            self.stop()

    def stop(self) -> None:
        if self._profiler is None or self._finished:
            return

        profiler_instance = self._profiler
        self._profiler = None
        self._finished = True

        profiler_instance.stop()


class NPUProfilerController:
    """Manage the automatically created process-level NPU profiler session."""

    def __init__(self):
        self._session: NPUProfilerSession | None = None
        self._finished = False
        self._exit_handler_registered = False

    def run(
            self,
            config: NPUProfileConfig,
            callback: Callable[[], T],
    ) -> T:
        """Run one user operation and advance the profiler once on success."""

        if not config.enabled or self._finished:
            return callback()

        if self._session is None and not self._start_session(config):
            return callback()

        try:
            result = callback()
        except BaseException:
            # Cleanup must not replace the user's original launch exception.
            self._close_quietly()
            raise

        session = self._session
        if session is None:
            return result

        try:
            session.step()
        except Exception:
            if config.strict:
                self._close_quietly()
                raise

            logger.warning(
                "NPU profiler step or stop failed; profiling is disabled",
                exc_info=True,
            )
            self._close_quietly()
        else:
            if session.finished:
                self._session = None
                self._finished = True

        return result

    def close(self) -> None:
        """Stop the current profiler session."""

        session = self._session
        self._session = None
        self._finished = True

        if session is not None:
            session.stop()

    def _start_session(self, config: NPUProfileConfig) -> bool:
        session = NPUProfilerSession(config)

        try:
            session.start()
        except Exception:
            if config.strict:
                raise

            logger.warning(
                "Failed to start NPU profiler; continuing without profiling",
                exc_info=True,
            )
            self._finished = True
            return False

        self._session = session

        if not self._exit_handler_registered:
            atexit.register(self._close_quietly)
            self._exit_handler_registered = True

        return True

    def _close_quietly(self) -> None:
        try:
            self.close()
        except Exception:
            logger.warning(
                "Failed to close NPU profiler",
                exc_info=True,
            )

_npu_profiler_controller = NPUProfilerController()

def get_npu_profiler_controller() -> NPUProfilerController:
    return _npu_profiler_controller

def _record_function(name: str):
    """Create a PyTorch annotation without importing torch_npu."""
    import torch

    return torch.profiler.record_function(name)


@contextmanager
def npu_annotation(
    name: str,
    *,
    enabled: bool = True,
    strict: bool = False,
) -> Iterator[None]:
    """Add a Timeline annotation around an existing operation."""

    if not enabled:
        yield
        return

    try:
        recorder = _record_function(name)
        recorder.__enter__()
    except Exception:
        if strict:
            raise

        logger.warning(
            "Failed to create NPU profiler annotation %r",
            name,
            exc_info=True,
        )
        yield
        return

    try:
        yield
    except BaseException:
        operation_error = sys.exc_info()

        try:
            recorder.__exit__(*operation_error)
        except Exception:
            logger.warning(
                "Failed to close NPU profiler annotation %r",
                name,
                exc_info=True,
            )

        # Always preserve the operator’s own exceptions.
        raise
    else:
        try:
            recorder.__exit__(None, None, None)
        except Exception:
            if strict:
                raise

            logger.warning(
                "Failed to close NPU profiler annotation %r",
                name,
                exc_info=True,
            )
