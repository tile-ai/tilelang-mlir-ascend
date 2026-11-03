"""Collect full-call NPU device intervals; external runner validates guard receipt.

Validate profiler configuration and raw record format against the task toolchain.
No NPU imports or initialization occur at module import time.
"""

import contextlib
import json
import statistics
from pathlib import Path

from profile_parser import _parse_ffts_profile


@contextlib.contextmanager
def capture_profile(directory):
    import torch_npu

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(directory.resolve()), analyse_flag=False
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=1, skip_first=0
        ),
        on_trace_ready=handler,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False, data_simplification=False,
        ),
    ) as prof:
        yield prof


def measure(call, *, output, flush, synchronize, frequency, check_reservation,
            verify_profile, protocol, expected_frequency, flush_bytes,
            warmup, repeats, flush_marker='_tilelang_profiler_cache_flush'):
    """Collect one case under an already-held device reservation.

    call(): one complete operator invocation, returning live output buffers.
    flush(): execute the declared flush_bytes cache flush outside the window.
    synchronize(): wait for this device's work to finish.
    frequency(): return actual protocol component MHz as a nonempty sequence.
    check_reservation(): raise unless this process owns its live reservation.
    verify_profile(path, parsed): raise on missing stages/initialization or
        unexpected measured work; inspect raw records, not just entry names.
    expected_frequency: frozen comparison MHz, or None to fix at first sample.
    protocol/warmup/repeats/flush_bytes: caller supplies the bound TASK values.

    The return is provisional until the external runner accepts the private
    guard receipt and guard/SSH/worker exits. No remote launch or retries here.
    """
    if warmup < 0 or repeats < 2 or flush_bytes <= 0 or not protocol:
        raise ValueError('invalid measurement protocol parameters')
    check_reservation()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    reference = tuple(expected_frequency) if expected_frequency is not None else None
    records = []
    phase = 'warmup'

    def observe(label):
        nonlocal reference
        values = tuple(frequency())
        if not values or any(isinstance(x, bool) or not isinstance(x, (int, float))
                             or not (0 < x < float('inf')) for x in values):
            raise RuntimeError('missing or invalid frequency readings')
        with (root / 'frequency.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase': label, 'mhz': values}) + '\n')
        if reference is None:
            reference = values
        if values != reference:
            raise RuntimeError(f'frequency mismatch: {values} != {reference}')

    try:
        for _ in range(warmup):
            check_reservation()
            flush()
            synchronize()
            result = call()
            synchronize()
            del result
        for index in range(repeats):
            phase = f'repeat-{index:02d}'
            check_reservation()
            flush()
            synchronize()
            observe(phase + ':before')
            with capture_profile(root / phase) as prof:
                result = call()
                synchronize()
                prof.step()
            del result
            observe(phase + ':after')
            raw = Path(prof.prof_if.prof_path)
            parsed = _parse_ffts_profile(raw, 1, exclude_name_marker=flush_marker)
            if (parsed['profile_excluded_kernel_count'] != 0
                    or parsed['profile_kernel_count'] <= 0
                    or parsed['latency_ms'] <= 0):
                raise RuntimeError('flush in sample or empty/nonpositive device activity')
            verify_profile(raw, parsed)
            record = {'sample': index, 'raw_profile': str(raw), 'parsed': parsed}
            records.append(record)
            with (root / 'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        values = [row['parsed']['latency_ms'] for row in records]
        summary = {
            'status': 'pending_guard_receipt', 'protocol': protocol,
            'metric': 'mean_device_busy_union_ms', 'warmup': warmup,
            'repeats': repeats, 'flush_bytes': flush_bytes,
            'frequency_mhz': reference, 'samples_ms': values,
            'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
            'min_ms': min(values), 'max_ms': max(values),
            'stdev_ms': statistics.stdev(values), 'records': records,
        }
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    except BaseException as error:
        (root / 'failure.json').write_text(json.dumps({
            'phase': phase, 'error_type': type(error).__name__, 'error': str(error),
            'completed_samples': len(records), 'performance_valid': False,
        }, indent=2) + '\n')
        raise
