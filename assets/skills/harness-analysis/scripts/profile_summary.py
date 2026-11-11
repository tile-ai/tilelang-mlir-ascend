"""Read explicitly scoped profiler exports; no device access or bottleneck verdicts.

Python standard library only. See references/device-profile.md for the manifest.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(data):
    def invalid_constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(data, parse_constant=invalid_constant)


def pointer(record, path):
    """JSON Pointer lookup, also usable for CSV headers containing punctuation."""
    require(isinstance(path, str) and path.startswith('/'), 'selector needs JSON Pointers')
    for key in path[1:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        if not isinstance(record, dict) or key not in record:
            return None
        record = record[key]
    return record


def matches(record, selector):
    return all(pointer(record, key) is not None and pointer(record, key) == value
               for key, value in selector.items())


def leaves(record, prefix=''):
    for key, value in record.items():
        path = prefix + '/' + key.replace('~', '~0').replace('/', '~1')
        if isinstance(value, dict):
            yield from leaves(value, path)
        else:
            yield path, value


def number(value):
    """No guessed units, percent scaling, thousands separator or missing-as-zero."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value):
            return None
    if not isinstance(value, (int, float, str)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def metric_stats(records, fields):
    """Only explicitly selected fields; never pool launches, passes or files."""
    result = {}
    for field in fields:
        samples = [(record['location'], number(pointer(record['raw'], field)))
                   for record in records]
        valid = [(loc, value) for loc, value in samples if value is not None]
        values = [value for _, value in valid]
        minimum, maximum = (min(values), max(values)) if values else (None, None)
        result[field] = {
            'numeric_count': len(values), 'missing_or_nonnumeric_count': len(samples) - len(values),
            'min': minimum, 'max': maximum,
            'mean': statistics.fmean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min_locations': [loc for loc, value in valid if value == minimum],
            'max_locations': [loc for loc, value in valid if value == maximum],
        }
    return result


def interval_stats(intervals):
    intervals = sorted(intervals)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy = sum(end - start for start, end in merged)
    span = merged[-1][1] - merged[0][0]
    return {'event_count': len(intervals),
            'interval_union': [[str(start), str(end)] for start, end in merged],
            'union_duration': float(busy), 'span': float(span),
            'uncovered_within_span': float(span - busy)}


def read_trace(data, source, kernel):
    payload = read_json(data)
    wrapped = isinstance(payload, dict)
    events = payload.get('traceEvents') if wrapped else payload
    require(isinstance(events, list), 'expected event array or object.traceEvents')
    records, tracks, unsupported = [], {}, []
    for index, event in enumerate(events):
        require(isinstance(event, dict), f'event {index} is not an object')
        if not matches(event, source['select']):
            continue
        location = ('/traceEvents' if wrapped else '') + f'/{index}'
        records.append({'location': location, 'raw': event})
        if event.get('ph') == 'X':
            start, duration = number(event.get('ts')), number(event.get('dur'))
            require(start is not None and duration is not None and duration >= 0,
                    f'{location}: invalid complete-event timestamp/duration')
            require(math.isfinite(start + duration), f'{location}: interval overflow')
            # Native exports use large decimal timestamp strings. Subtract before
            # converting to float so sub-microsecond durations are not rounded away.
            start, duration = Decimal(str(event['ts'])), Decimal(str(event['dur']))
            key = tuple(str(event.get(k, 'unknown')) for k in ('pid', 'tid', 'cat', 'name'))
            tracks.setdefault(key, []).append((start, start + duration))
        elif event.get('ph') not in ('C', 'i', 'I', 'M'):
            unsupported.append(location)
    if source['role'] == 'kernel':
        require(source['select'].get('/name') == kernel,
                'kernel trace selector must match identity.kernel exactly')
        require(len(records) == 1 and records[0]['raw'].get('ph') == 'X',
                'kernel trace must select exactly one complete event; refine stream/task/ts selectors')
    return records, {'tracks': [dict(zip(('pid', 'tid', 'cat', 'name'), key),
                                    **interval_stats(intervals)) for key, intervals in tracks.items()],
                     'unsupported_event_locations': unsupported,
                     'time_unit': source.get('time_unit', 'unknown'),
                     'time_unit_basis': 'declared in input; not inferred from displayTimeUnit'}


def read_csv(data, source, kernel):
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    original_headers = next(reader, [])
    headers = list(original_headers)
    while headers and headers[-1] == '':
        headers.pop()
    require(headers and len(headers) == len(set(headers)) and all(headers),
            'CSV needs unique nonempty headers')
    records = []
    for index, cells in enumerate(reader, 1):
        require(len(cells) in (len(headers), len(original_headers)) and
                all(value == '' for value in cells[len(headers):]),
                f'CSV record {index}: column count mismatch')
        row = dict(zip(headers, cells))
        if matches(row, source['select']):
            records.append({'location': f'record:{index}', 'line_end': reader.line_num, 'raw': row})
    if source['role'] == 'kernel':
        field = source.get('kernel_field', '/Op Name')
        pointer({}, field)
        require(source['select'].get(field) == kernel,
                'kernel CSV selector must match identity.kernel exactly')
        require(len(records) == 1, 'kernel CSV must select exactly one row; refine launch selectors')
    return records, {'headers': headers,
                     'trailing_empty_columns_ignored': len(original_headers) - len(headers)}


def summarize(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = read_json(manifest_bytes)
    require(isinstance(manifest, dict) and manifest.get('schema') == 'operator-harness/profile-input@1',
            'expected operator-harness/profile-input@1')
    identity = manifest.get('identity')
    require(isinstance(identity, dict), 'identity must be an object')
    for key in ('project_id', 'run_id', 'candidate', 'case', 'build', 'environment', 'kernel'):
        require(isinstance(identity.get(key), str) and identity[key].strip(), f'missing identity.{key}')
    sources = manifest.get('sources')
    require(isinstance(sources, list) and sources, 'sources must be a nonempty list')
    report = {'schema': 'operator-harness/profile-summary@1', 'purpose': 'analysis_only',
              'input_sha256': digest(manifest_bytes), 'parser_sha256': digest(Path(__file__).read_bytes()),
              'identity': identity, 'identity_status': 'declared; reconcile with TASK and ANALYSIS_DATA',
              'status': 'parsed', 'sources': []}
    ids = set()
    paths = set()
    for source in sources:
        require(isinstance(source, dict), 'source must be an object')
        source_id = source.get('id')
        require(isinstance(source_id, str) and source_id and source_id not in ids, 'source IDs must be unique')
        ids.add(source_id)
        result = {'id': source_id, 'path': source.get('path'), 'format': source.get('format'),
                  'pass_id': source.get('pass_id'),
                  'tool': source.get('tool'), 'status': 'parsed', 'records': []}
        report['sources'].append(result)
        try:
            for field in ('path', 'pass_id', 'tool', 'scope_evidence'):
                require(isinstance(source.get(field), str) and source[field].strip(), f'missing source.{field}')
            require(source.get('role') in ('kernel', 'metrics', 'timeline'), 'unsupported source.role')
            require(isinstance(source.get('select'), dict), 'select must be an object')
            for key, value in source['select'].items():
                pointer({}, key)
                require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
                        'selectors use exact strings or numbers')
            fields = source.get('metric_fields', [])
            require(isinstance(fields, list) and all(isinstance(v, str) for v in fields),
                    'metric_fields must be a list of JSON Pointers')
            for field in fields:
                pointer({}, field)
            rel = Path(source['path'])
            require(not rel.is_absolute() and '..' not in rel.parts,
                    'source path must be relative to manifest directory')
            raw_path = manifest_path.parent / rel
            require(not any(p.is_symlink() for p in (raw_path, *raw_path.parents)
                            if p != manifest_path.parent and p.is_relative_to(manifest_path.parent)),
                    'source path must not traverse symlinks')
            path = raw_path.resolve()
            require(path.is_relative_to(manifest_path.parent), 'source path escapes manifest directory')
            require(path not in paths, 'duplicate input file; use a separate manifest for another launch')
            paths.add(path)
            data = path.read_bytes()
            result['sha256'] = digest(data)
            require(result['sha256'] == source.get('sha256'), 'input SHA256 mismatch')
            result.update({key: source[key] for key in ('role', 'select', 'scope_evidence')})
            if source.get('format') == 'trace':
                records, details = read_trace(data, source, identity['kernel'])
            elif source.get('format') == 'csv':
                records, details = read_csv(data, source, identity['kernel'])
            else:
                raise ValueError('unsupported format; export CSV/Chrome trace JSON with the matching official tool')
            require(records, 'no matching records')
            result.update(details)
            result['records'] = records
            result['available_fields'] = sorted({field for rec in records for field, _ in leaves(rec['raw'])})
            result['metrics'] = metric_stats(records, fields)
            if details.get('unsupported_event_locations'):
                result['status'] = 'partial'
            if any(v['numeric_count'] == 0 for v in result['metrics'].values()):
                result['status'] = 'partial'
        except (ValueError, OSError, UnicodeError, csv.Error) as exc:
            result['status'] = 'error'
            result['error'] = str(exc)
        if result['status'] != 'parsed':
            report['status'] = 'partial'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='profile-input@1 JSON manifest')
    parser.add_argument('--output', required=True, type=Path, help='new summary JSON; never overwritten')
    args = parser.parse_args(argv)
    try:
        report = summarize(args.input)
        output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(output)
        print(json.dumps({'status': report['status'], 'output': str(args.output)}, ensure_ascii=False))
        return 0 if report['status'] == 'parsed' else 2
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
