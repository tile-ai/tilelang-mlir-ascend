#!/usr/bin/env python3
"""Conditional Cube-only roofline from explicit resource demands; Python 3.9+."""
import argparse
import json
import math
from pathlib import Path
from calculate_mfu import CATALOG, integer


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + ' must be finite and positive')
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ' must be nonempty text')
    return value


def fields(obj, required, optional=()):
    if not isinstance(obj, dict):
        raise ValueError('expected an object')
    if set(required) - obj.keys():
        raise ValueError('missing fields: ' + str(sorted(set(required) - obj.keys())))
    if obj.keys() - set(required) - set(optional):
        raise ValueError('unknown fields: ' + str(sorted(obj.keys() - set(required) - set(optional))))


def calculate(spec):
    fields(spec, ('device', 'device_cores', 'cores', 'dtype', 'useful_ops', 'executed_ops', 'resources', 'assumptions'),
           ('measured_time_us', 'label'))
    total = integer(spec['device_cores'], 'device_cores', 1)
    cores = integer(spec['cores'], 'cores', 1)
    if cores > total:
        raise ValueError('cores exceeds device_cores')
    if spec['dtype'] == 'int8':
        raise ValueError('this roofline tool reports floating-point MFU, not INT8 utilization')
    if not isinstance(spec['device'], str) or not isinstance(spec['dtype'], str):
        raise ValueError('device and dtype must be strings')
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    try:
        peak = catalog['devices'][spec['device']][str(total)][spec['dtype']] * cores / total
    except KeyError:
        raise ValueError('unverified device/core-count/dtype combination') from None
    useful = integer(spec['useful_ops'], 'useful_ops')
    executed = integer(spec['executed_ops'], 'executed_ops', 1)
    if useful > executed:
        raise ValueError('useful_ops exceeds executed_ops')
    assumptions = spec['assumptions']
    if not isinstance(assumptions, list) or not assumptions:
        raise ValueError('assumptions must be a nonempty list')
    for item in assumptions:
        text(item, 'assumption')
    if not isinstance(spec['resources'], list):
        raise ValueError('resources must be a list')
    compute_seconds = executed / (peak * 1e12)
    constraints = {'cube': compute_seconds}
    details = []
    for resource in spec['resources']:
        fields(resource, ('name', 'payload_bytes', 'bandwidth_gbps', 'basis', 'source'))
        name = text(resource['name'], 'resource name')
        if name in constraints:
            raise ValueError('duplicate/reserved resource name: ' + name)
        if resource['basis'] not in ('official', 'measured', 'assumed'):
            raise ValueError('basis must be official, measured, or assumed')
        text(resource['source'], 'bandwidth source')
        flows = resource['payload_bytes']
        if not isinstance(flows, dict) or not flows:
            raise ValueError('payload_bytes must be a nonempty flow-name to byte-count object')
        for key, count in flows.items():
            text(key, 'flow name')
            integer(count, 'payload bytes')
        count = sum(flows.values())
        bandwidth = positive(resource['bandwidth_gbps'], 'bandwidth_gbps') * 1e9
        seconds = count / bandwidth
        constraints[name] = seconds
        details.append(dict(**resource, total_bytes=count, time_floor_us=seconds*1e6,
                            useful_flops_per_byte=useful/count if count else None,
                            standalone_useful_tflops_ceiling=useful/seconds/1e12 if seconds else None))
    seconds = max(constraints.values())
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('time floor out of range')
    result = dict(input=spec, metric='conditional Cube-only roofline', used_cube_peak_tflops=peak,
                  useful_fraction=useful/executed, compute_floor_us=compute_seconds*1e6,
                  compute_only_mfu_ceiling_percent=100*useful/executed,
                  useful_compute_roof_tflops=peak*useful/executed,
                  time_floor_us=seconds*1e6, useful_tflops_ceiling=useful/seconds/1e12,
                  mfu_ceiling_percent=100*useful/(seconds*peak*1e12),
                  limiting_resources=[k for k,v in constraints.items() if math.isclose(v,seconds,rel_tol=1e-12,abs_tol=0)],
                  resources=details, measured_mfu_percent=None,
                  cube_peak_source=catalog['source_url'], cube_peak_checked_on=catalog['checked_on'],
                  warnings=['Conditional bound with ideal overlap of the listed resources; omitted constraints can lower achievable performance.'])
    if 'measured_time_us' in spec:
        measured = positive(spec['measured_time_us'], 'measured_time_us') * 1e-6
        result['measured_mfu_percent'] = 100*useful/(measured*peak*1e12)
        if measured < seconds:
            result['warnings'].append('Measurement is faster than this scenario floor: check cache traffic, rates and timing scope; do not clamp MFU or infer bandwidth from the same time.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = calculate(json.loads(args.input.read_text(encoding='utf-8-sig')))
        output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(output, encoding='utf-8')
        else:
            print(output, end='')
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
