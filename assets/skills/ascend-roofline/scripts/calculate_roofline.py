#!/usr/bin/env python3
"""Conditional Cube/Vector roofline with explicit resource demands; Python 3.9+."""
import argparse
import json
import math
from pathlib import Path
from calculate_mfu import integer
from peak_catalog import select_peak


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
    fields(spec, ('dtype', 'useful_ops', 'executed_ops', 'resources', 'assumptions'),
           ('engine', 'device', 'device_cores', 'cores', 'peak_tops', 'peak_source', 'measured_time_us', 'label'))
    engine = spec.get('engine', 'cube')
    if engine not in ('cube', 'vector'):
        raise ValueError('engine must be cube or vector')
    dtype = text(spec['dtype'], 'dtype')
    floating = not dtype.lower().startswith(('int', 'uint'))
    if spec.get('peak_tops') is not None:
        if any(spec.get(key) is not None for key in ('device', 'device_cores', 'cores')):
            raise ValueError('explicit peak and catalog selection are mutually exclusive')
        peak = positive(spec['peak_tops'], 'peak_tops')
        provenance = dict(source=text(spec.get('peak_source'), 'peak_source'), checked_on=None)
    else:
        if spec.get('peak_source') is not None:
            raise ValueError('peak_source requires peak_tops')
        if any(spec.get(key) is None for key in ('device', 'device_cores', 'cores')):
            raise ValueError('provide device, device_cores and cores, or peak_tops with peak_source')
        provenance = select_peak(spec['device'],spec['device_cores'],spec['cores'],dtype,engine)
        peak = provenance['peak_tops']
    useful = integer(spec['useful_ops'], 'useful_ops')
    executed = integer(spec['executed_ops'], 'executed_ops')
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
    constraints = {engine: compute_seconds}
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
                            useful_ops_per_byte=useful/count if count else None,
                            executed_ops_per_byte=executed/count if count else None,
                            standalone_useful_tops_ceiling=useful/seconds/1e12 if seconds else None))
    seconds = max(constraints.values())
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('time floor out of range')
    fraction = useful/executed if executed else None
    utilization = 100*useful/(seconds*peak*1e12) if executed else None
    result = dict(input=spec, metric=f'conditional {engine} roofline', engine=engine, used_peak_tops=peak,
                  peak_unit='TFLOPS' if floating else 'TOPS',
                  useful_fraction=fraction, compute_floor_us=compute_seconds*1e6,
                  compute_only_utilization_ceiling_percent=100*fraction if fraction is not None else None,
                  compute_only_mfu_ceiling_percent=100*fraction if floating and fraction is not None else None,
                  useful_compute_roof_tops=peak*fraction if fraction is not None else None,
                  time_floor_us=seconds*1e6, useful_tops_ceiling=useful/seconds/1e12,
                  utilization_ceiling_percent=utilization,
                  mfu_ceiling_percent=utilization if floating else None,
                  limiting_resources=[k for k,v in constraints.items() if math.isclose(v,seconds,rel_tol=1e-12,abs_tol=0)],
                  resources=details, measured_utilization_percent=None, measured_mfu_percent=None,
                  peak_source=provenance['source'], peak_selection=provenance,
                  warnings=['Conditional bound with ideal overlap of the listed resources; omitted constraints can lower achievable performance.'])
    if 'measured_time_us' in spec:
        measured = positive(spec['measured_time_us'], 'measured_time_us') * 1e-6
        result['measured_utilization_percent'] = 100*useful/(measured*peak*1e12) if executed else None
        result['measured_mfu_percent'] = result['measured_utilization_percent'] if floating and executed else None
        result['measured_useful_tops'] = useful/measured/1e12
        result['measured_executed_tops'] = executed/measured/1e12
        for resource in details:
            resource['achieved_gbps'] = resource['total_bytes']/measured/1e9
            resource['bwu_percent'] = 100*resource['achieved_gbps']/resource['bandwidth_gbps']
        if measured < seconds:
            result['warnings'].append('Measurement is faster than this scenario floor: check cache traffic, rates and timing scope; do not clamp MFU or infer bandwidth from the same time.')
    # Preserve the former Cube floating-point output fields for existing consumers.
    if engine == 'cube' and floating:
        result.update(used_cube_peak_tflops=peak,cube_peak_source=provenance['source'],
                      cube_peak_checked_on=provenance['checked_on'],
                      useful_compute_roof_tflops=result['useful_compute_roof_tops'],
                      useful_tflops_ceiling=result['useful_tops_ceiling'])
        for resource in details:
            resource['useful_flops_per_byte']=resource['useful_ops_per_byte']
            resource['standalone_useful_tflops_ceiling']=resource['standalone_useful_tops_ceiling']
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
