"""Diagnostic collection helpers, separate from benchmark acceptance.

Use msprof_command for full multi-metric collection: msOpProf owns event groups,
kernel/application replay, instrumentation, and driver collection. Native capture
only supplements call mapping and basic counters; it is not a required first pass.
No NPU imports at module import time. The bound runner owns device reservations,
process exits, timeouts, and receipts. Callers review kernel/metric coverage before
promoting the generated artifact index to ANALYSIS_DATA.md.
"""

import hashlib
import json
from pathlib import Path
import re
import emission


OP_METRICS = ("Default", "PipeTimeline", "Occupancy", "Roofline", "Memory")
TORCH_METRICS = ("PipeUtilization", "ArithmeticUtilization", "Memory", "MemoryL0",
                 "MemoryUB", "ResourceConflictRatio", "L2Cache", "MemoryAccess")


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,1024}", value):
        raise ValueError("use a recorded kernel/range name, without wildcards or shell syntax")
    return value


def msprof_command(application, *, kernel_name, output, range_name=None, launch_count=1,
                   metrics=OP_METRICS, tool=("msprof", "op"), replay_mode="kernel",
                   launch_skip_before_match=0, mstx_value="on"):
    """Return argv, never a shell string or an execution/permission grant.

    Verify tool help and chip support before calling. Kernel-name filtering is
    prefix-based in msOpProf; actual collected kernels still require inspection.
    MSTX is optional and needs a verified torch_npu/collector binding. Select
    mstx_value="on" or "true" from the installed tool's help; releases use
    different boolean CLI spellings. Skip counts all launches before filtering,
    not just launches matching kernel_name.
    Keep the metric combination in one invocation; metric count is not replay
    count. The collector backs up/restores tracked device memory during ordinary
    kernel replay. Verify that coverage for the target's state; Python prepare
    is not invoked between internal replays. Application replay reruns the program.
    This helper intentionally excludes range replay, which conflicts with several
    requested timeline/roofline metrics on relevant toolchains.
    """
    if (not isinstance(application, (list, tuple)) or not application
            or any(not isinstance(x, str) or not x or "\0" in x for x in application)):
        raise ValueError("application must be a nonempty argv list")
    if tuple(tool) not in (("msprof", "op"), ("msopprof",)):
        raise ValueError("select the tool entry verified in the bound environment")
    if type(launch_count) is not int or not 1 <= launch_count <= 5000:
        raise ValueError("invalid launch_count")
    if type(launch_skip_before_match) is not int or not 0 <= launch_skip_before_match <= 1000:
        raise ValueError("invalid launch_skip_before_match")
    if replay_mode not in ("kernel", "application"):
        raise ValueError("unsupported replay mode for this diagnostic recipe")
    if mstx_value not in ("on", "true"):
        raise ValueError("select the enabled MSTX spelling from installed tool help")
    if (not isinstance(metrics, (list, tuple)) or not metrics
            or len(set(metrics)) != len(metrics) or any(x not in OP_METRICS for x in metrics)):
        raise ValueError("select supported metrics from the documented diagnostic recipe")
    selection = [] if range_name is None else ["--mstx=" + mstx_value,
                                               "--mstx-include=" + _name(range_name)]
    return [*tool, "--aic-metrics=" + ",".join(metrics),
            "--kernel-name=" + _name(kernel_name), *selection,
            f"--launch-count={launch_count}",
            f"--launch-skip-before-match={launch_skip_before_match}",
            "--replay-mode=" + replay_mode, "--output=" + str(output), *application]


def marked_call(call, *, prepare, synchronize, check_reservation, range_name, warmup):
    """Run one MSTX host range under an external msprof op collector.

    prepare() compiles/allocates/resets outside the range; it must restore the same
    input state before every call. synchronize() must cover all participating
    streams. Do not enable torch_npu.profiler.profile in this process mode.
    MSTX's host range selects launches, not a device-time measurement. The kernel
    return value stays live until synchronization finishes.
    """
    _name(range_name)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    check_reservation()
    import torch_npu

    for _ in range(warmup):
        check_reservation()
        prepare()
        synchronize()
        result = call()
        synchronize()
        del result
    check_reservation()
    prepare()
    synchronize()
    range_id = torch_npu.npu.mstx.range_start(range_name)
    if type(range_id) is not int or range_id <= 0:
        raise RuntimeError("MSTX range did not start; check the external collector and binding")
    try:
        result = call()
        synchronize()
        return result
    finally:
        torch_npu.npu.mstx.range_end(range_id)


def artifact_index(directory, workspace_root=None):
    """List files with hashes, without interpreting proprietary binary formats."""
    root = Path(directory).resolve()
    if workspace_root is not None:
        return {"schema": "operator-harness/artifact-index@1", "files": emission.files(workspace_root, root)}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("diagnostic artifacts must not be symlinks")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": digest.hexdigest()})
    return files


def capture_diagnostics(call, *, output, identity, metrics, prepare, synchronize,
                        check_reservation, warmup, evidence_context=None):
    """Optionally supplement mapping/counters with separate native profiler passes.

    identity records candidate, case, build and environment IDs/digests. Each pass
    has one full operator call after prepare/reset; no metrics are silently
    substituted. Output is pending review, not a performance score or validated
    pipeline/occupancy/roofline capture. This is a separate process mode from
    external msprof op. Unsupported API/export failures propagate with evidence.
    """
    if any(not isinstance(identity.get(key), str) or not identity[key].strip()
           for key in ("candidate", "case", "build", "environment")):
        raise ValueError("diagnostics require candidate/case/build/environment identity")
    if (not isinstance(metrics, (list, tuple)) or not metrics or len(set(metrics)) != len(metrics)
            or any(name not in TORCH_METRICS for name in metrics)):
        raise ValueError("native torch_npu accepts one supported enum per pass, not msprof op metrics")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    check_reservation()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    report = {"schema": "operator-harness/torch-diagnostics@1", "status": "running",
              "identity": dict(identity), "requested_metrics": list(metrics), "warmup": warmup,
              "purpose": "analysis_only", "passes": []}

    def save():
        (root / "collection.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import torch_npu
        profiler = torch_npu.profiler
        report["torch_npu_version"] = str(getattr(torch_npu, "__version__", "unknown"))
        values = {name: getattr(profiler.AiCMetrics, name) for name in metrics}
        supported = profiler.supported_ai_core_metrics()
        if any(value not in supported for value in values.values()):
            raise RuntimeError("requested metric not supported by installed torch_npu")
        for _ in range(warmup):
            check_reservation()
            prepare()
            synchronize()
            result = call()
            synchronize()
            del result
        for name, value in values.items():
            check_reservation()
            prepare()
            synchronize()
            pass_dir = root / name
            pass_dir.mkdir()
            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=profiler.tensorboard_trace_handler(
                    str(pass_dir.resolve()), analyse_flag=True, async_mode=False),
                experimental_config=profiler._ExperimentalConfig(
                    profiler_level=profiler.ProfilerLevel.Level1, aic_metrics=value,
                    export_type=profiler.ExportType.Text, data_simplification=False),
            ) as prof:
                result = call()
                synchronize()
                prof.step()
            del result
            files = artifact_index(pass_dir)
            if not files or not any(row["bytes"] for row in files):
                raise RuntimeError(f"no diagnostic artifacts exported for {name}")
            report["passes"].append({"metric": name, "directory": name, "files": files})
            save()
        report["status"] = "pending_review"
        save()
        if evidence_context is not None:
            report["evidence"] = emission.diagnostics(evidence_context, root / "diagnostic-evidence.json",
                emission.files(evidence_context["workspace_root"], root))
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise
