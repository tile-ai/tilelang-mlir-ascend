#!/usr/bin/env python3
"""Calculate bandwidth from explicitly selected CSV bytes and a matching duration."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path

UNITS = {'B': 1, 'KB': 1000, 'KiB': 1024, 'MB': 10**6, 'MiB': 2**20}


def number(value, label, positive=False):
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f'{label}: expected a number, not NA or missing data') from None
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError(f'{label}: expected a finite {"positive" if positive else "nonnegative"} number')
    return result


def calculate(args):
    if len(set(args.row)) != len(args.row) or any(row < 1 for row in args.row):
        raise ValueError('rows must be distinct positive 1-based data row numbers')
    if len(set(args.column)) != len(args.column):
        raise ValueError('duplicate columns would double-count traffic')
    if (args.peak_gbps is None) != (args.peak_source is None):
        raise ValueError('--peak-gbps and --peak-source must be supplied together')
    for name in ('resource', 'time_source'):
        if not getattr(args, name).strip():
            raise ValueError(f'{name} must not be empty')
    if args.peak_source is not None and not args.peak_source.strip():
        raise ValueError('peak_source must not be empty')
    raw = args.csv.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    headers = reader.fieldnames or []
    if len(set(headers)) != len(headers):
        raise ValueError('duplicate CSV headers are ambiguous')
    if not set(args.column).issubset(headers):
        raise ValueError('requested column is missing from CSV')
    selected = []
    total = Decimal(0)
    wanted = set(args.row)
    for index, row in enumerate(reader, 1):
        if index not in wanted:
            continue
        values = {}
        for column in args.column:
            value = number(row[column], f'row {index}, column {column}')
            total += value * UNITS[args.amount_unit]
            values[column] = row[column]
        selected.append({'row': index, 'values': values})
    if len(selected) != len(wanted):
        raise ValueError('requested row is missing from CSV')
    seconds = number(args.time_us, 'time_us', positive=True) * Decimal('1e-6')
    bandwidth = total / seconds / Decimal(10**9)
    peak = None if args.peak_gbps is None else number(args.peak_gbps, 'peak_gbps', positive=True)
    utilization = None if peak is None else 100 * bandwidth / peak
    for value in (total, seconds, bandwidth, peak, utilization):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError('result exceeds supported output range')
    return {
        'csv': str(args.csv.resolve()), 'csv_sha256': hashlib.sha256(raw).hexdigest(),
        'selected': selected, 'resource': args.resource, 'direction': args.direction,
        'amount_unit': args.amount_unit, 'total_bytes': float(total),
        'time_seconds': float(seconds), 'time_scope': args.time_scope,
        'time_source': args.time_source, 'achieved_gbps': float(bandwidth),
        'peak_gbps': None if peak is None else float(peak), 'peak_source': args.peak_source,
        'bwu_percent': None if utilization is None else float(utilization),
        'warnings': ['Above 100%: check units, row scope, timing, direction and peak applicability.']
        if utilization is not None and utilization > 100 else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--row', type=int, action='append', required=True,
                        help='1-based data row, excluding header; repeat for disjoint records of ONE launch')
    parser.add_argument('--column', action='append', required=True,
                        help='byte-amount column, not a rate/percentage; repeat for disjoint traffic')
    parser.add_argument('--amount-unit', choices=UNITS, required=True,
                        help='verified unit, which may differ from the CSV header')
    parser.add_argument('--resource', required=True, help='e.g. HBM, L2-to-UB, MTE2 GM-to-UB')
    parser.add_argument('--direction', choices=('read', 'write', 'read+write'), required=True)
    parser.add_argument('--time-us', required=True)
    parser.add_argument('--time-scope', choices=('task', 'active'), required=True)
    parser.add_argument('--time-source', required=True, help='matching task or pipeline duration evidence')
    parser.add_argument('--peak-gbps', help='matching resource peak in decimal GB/s; no default')
    parser.add_argument('--peak-source', help='peak evidence, hardware/scope and direction')
    args = parser.parse_args()
    try:
        result = calculate(args)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    except (ValueError, OSError, OverflowError, csv.Error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
