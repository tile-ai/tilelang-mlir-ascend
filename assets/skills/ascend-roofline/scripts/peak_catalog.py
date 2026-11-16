"""Shared, explicitly scoped Cube/Vector peak selection."""
import json
import math
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / 'assets/peaks.json'


def select_peak(device, device_cores, cores, dtype, engine='cube'):
    if engine not in ('cube', 'vector'):
        raise ValueError('engine must be cube or vector')
    if any(type(x) is not int or x < 1 for x in (device_cores, cores)):
        raise ValueError('device_cores and cores must be positive integers')
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    try:
        variant = catalog['devices'][device][str(device_cores)]
        total = variant['aic_cores' if engine == 'cube' else 'aiv_cores']
        peak = variant[engine][dtype]
    except (KeyError, TypeError):
        raise ValueError('unverified device/core-count/engine/dtype combination') from None
    if cores > total:
        raise ValueError('participating cores exceed physical engine cores')
    return dict(peak_tops=peak * cores / total, device_peak=peak,
                device_engine_cores=total, source=catalog['source_url'],
                checked_on=catalog['checked_on'], source_table=catalog['source_table'])


def counted_utilization(engine, dtype, useful_ops, time, unit, iterations, peak, source):
    if engine not in ('cube', 'vector') or not isinstance(dtype, str) or not dtype.strip():
        raise ValueError('valid engine and nonempty dtype required')
    if type(useful_ops) is not int or useful_ops < 0:
        raise ValueError('useful_ops must be a nonnegative integer')
    if type(iterations) is not int or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    for value in (time, peak):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('time and peak must be finite and positive')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('peak source must be nonempty text')
    if unit not in ('s', 'ms', 'us', 'ns'):
        raise ValueError('unsupported time unit')
    seconds = time * {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}[unit] / iterations
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('normalized duration out of range')
    achieved = useful_ops / seconds / 1e12
    utilization = achieved / peak * 100
    if not math.isfinite(utilization):
        raise ValueError('utilization out of range')
    floating = not dtype.lower().startswith(('int', 'uint'))
    return dict(metric=f'{engine} MFU' if floating else f'{engine} integer utilization',
                engine=engine, dtype=dtype, useful_ops_per_iteration=useful_ops,
                seconds_per_iteration=seconds, achieved_teraops_per_second=achieved,
                used_peak=peak, peak_unit='TFLOPS' if floating else 'TOPS',
                utilization_percent=utilization,
                mfu_percent=utilization if floating else None,
                peak_source=source,
                warnings=['Above 100%: check work, time, units, scope and peak.'] if utilization > 100 else [])
