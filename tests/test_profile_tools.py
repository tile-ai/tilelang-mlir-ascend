"""CPU-only checks for the measured profiler format and collection protocol."""

import contextlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'assets/skills/harness-device-test/scripts'))
from profile_parser import _parse_ffts_profile
import torch_npu_measure


def record(start, end, seq=1):
    values = [0] * 16
    values[0], values[1], values[2] = seq << 32, 10, 1 << 31
    values[14], values[15] = start, end
    return struct.pack('<16q', *values)


class ProfileToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_union_excludes_host_gaps_and_counts_overlap_once(self):
        (self.root / 'ffts_profile.slice_0').write_bytes(
            record(1000, 4000) + record(3000, 5000, 2) + record(9000, 10000, 3))
        result = _parse_ffts_profile(self.root, 1)
        self.assertAlmostEqual(result['latency_ms'], .005)
        self.assertAlmostEqual(result['kernel_span_ms'], .009)
        self.assertAlmostEqual(result['profile_kernel_us'], 6)
        self.assertEqual(result['profile_kernel_count'], 3)

    def test_empty_truncated_and_negative_records_fail(self):
        path = self.root / 'ffts_profile.slice_0'
        for payload in (b'', record(1, 2) + b'x', record(2, 1)):
            with self.subTest(payload_size=len(payload)):
                path.write_bytes(payload)
                with self.assertRaises(RuntimeError):
                    _parse_ffts_profile(self.root, 1)

    def test_named_flush_is_excluded_and_reported(self):
        (self.root / 'hash_dic.slice_0').write_text('42:cache_flush\n43:target\n')
        tracks = []
        for seq, name_hash in ((1, 42), (2, 43)):
            values = [0] * 8
            values[3], values[5] = seq << 32, name_hash
            tracks.append(struct.pack('<8q', *values))
        (self.root / 'task_track.slice_0').write_bytes(b''.join(tracks))
        (self.root / 'ffts_profile.slice_0').write_bytes(
            record(0, 1000) + record(2000, 4000, 2))
        result = _parse_ffts_profile(self.root, 1, exclude_name_marker='cache_flush',
                                     require_marker_seen=True)
        self.assertAlmostEqual(result['latency_ms'], .002)
        self.assertEqual(result['profile_excluded_kernel_count'], 1)

    def measure(self, readings, verify=None):
        events = []
        readings = iter(readings)

        @contextlib.contextmanager
        def capture(directory):
            directory.mkdir()
            (directory / 'ffts_profile.slice_0').write_bytes(record(0, 1000))
            yield SimpleNamespace(prof_if=SimpleNamespace(prof_path=directory),
                                  step=lambda: events.append('step'))

        with patch.object(torch_npu_measure, 'capture_profile', capture):
            result = torch_npu_measure.measure(
                lambda: events.append('call'), output=self.root / 'measurement',
                flush=lambda: events.append('flush'),
                synchronize=lambda: events.append('sync'),
                frequency=lambda: next(readings),
                check_reservation=lambda: events.append('reservation'),
                verify_profile=verify or (lambda path, parsed: None),
                protocol='test-v1', expected_frequency=(1650, 1650),
                flush_bytes=1024, warmup=1, repeats=2)
        return result, events

    def test_success_stays_provisional_and_preserves_samples(self):
        result, events = self.measure([(1650, 1650)] * 4)
        self.assertEqual(result['status'], 'pending_guard_receipt')
        self.assertEqual(result['samples_ms'], [.001, .001])
        self.assertEqual(events.count('call'), 3)
        self.assertEqual(events.count('flush'), 3)
        self.assertEqual(events.count('step'), 2)
        self.assertEqual(len((self.root / 'measurement/samples.jsonl')
                             .read_text().splitlines()), 2)

    def test_frequency_drift_preserves_failure_without_success_summary(self):
        with self.assertRaisesRegex(RuntimeError, 'frequency mismatch'):
            self.measure([(1650, 1650), (100, 1650)])
        failure = json.loads((self.root / 'measurement/failure.json').read_text())
        self.assertFalse(failure['performance_valid'])
        self.assertEqual(failure['completed_samples'], 0)
        self.assertFalse((self.root / 'measurement/summary.json').exists())

    def test_missing_stage_is_not_accepted_as_performance(self):
        def reject(path, parsed):
            raise RuntimeError('missing operator stage')

        with self.assertRaisesRegex(RuntimeError, 'missing operator stage'):
            self.measure([(1650, 1650)] * 2, verify=reject)
        self.assertFalse((self.root / 'measurement/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
