"""Offline diagnostics must preserve provenance and avoid invented measurements."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'assets/skills/harness-analysis/scripts/profile_summary.py'
SPEC = importlib.util.spec_from_file_location('analysis_profile_summary', SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class AnalysisToolsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = {
            'schema': 'operator-harness/profile-input@1',
            'identity': dict(project_id='example', run_id='a1', search_node_id='N0',
                             candidate='C1', case='large', build='build1',
                             environment='env1', kernel='target'),
            'sources': [],
        }

    def add(self, name, content, fmt='trace', role='kernel', select=None, fields=None, pass_id='p1'):
        path = self.root / name
        path.write_text(content, encoding='utf-8')
        source = {'id': name, 'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'format': fmt, 'role': role, 'pass_id': pass_id, 'tool': 'synthetic fixture v1',
                  'scope_evidence': 'synthetic single launch with separate PMU passes',
                  'select': {'/name': 'target'} if select is None else select,
                  'metric_fields': fields or []}
        self.manifest['sources'].append(source)
        return source

    def run_summary(self):
        path = self.root / 'input.json'
        path.write_text(json.dumps(self.manifest))
        return summary.summarize(path)

    @staticmethod
    def event(start=0, duration=10, name='target', **kwargs):
        return dict(name=name, ph='X', ts=start, dur=duration, pid=1, tid=2, **kwargs)

    def test_array_and_object_trace_preserve_nested_metrics_and_locations(self):
        event = self.event(args={'aicMetric': {'vec_ratio': 0.7, 'mte_ratio': 'N/A'}})
        for wrapped in (False, True):
            self.manifest['sources'] = []
            payload = {'traceEvents': [event], 'displayTimeUnit': 'ns'} if wrapped else [event]
            self.add('trace.json', json.dumps(payload), fields=['/args/aicMetric/vec_ratio'])
            report = self.run_summary()
            self.assertEqual(report['status'], 'parsed')
            source = report['sources'][0]
            self.assertEqual(source['metrics']['/args/aicMetric/vec_ratio']['mean'], .7)
            self.assertEqual(source['records'][0]['location'], '/traceEvents/0' if wrapped else '/0')
            self.assertEqual(source['records'][0]['raw'], event)
            self.assertEqual(source['time_unit'], 'unknown')

    def test_same_name_launches_require_exact_instance(self):
        events = [self.event(args={'Task Id': 1}), self.event(20, args={'Task Id': 2}),
                  self.event(name='target_helper')]
        source = self.add('trace.json', json.dumps(events))
        self.assertEqual(self.run_summary()['status'], 'partial')
        source['select']['/args/Task Id'] = 2
        result = self.run_summary()['sources'][0]
        self.assertEqual(result['status'], 'parsed')
        self.assertEqual(result['records'][0]['location'], '/1')

    def test_csv_keeps_aic_aiv_missing_values_and_raw_units_separate(self):
        content = '\ufeffblock_id,sub_block_id,aic_time(us),aiv_time(us),ratio(%),bw(GB/s)\n'
        content += '0,AIC,4,N/A,50%,12\n0,AIV0,N/A,8,0,8\n1,AIV0,N/A,0,,NaN\n'
        self.add('Pipe.csv', content, fmt='csv', role='metrics', select={},
                 fields=['/aic_time(us)', '/aiv_time(us)', '/ratio(%)', '/bw(GB~1s)'])
        result = self.run_summary()['sources'][0]
        self.assertEqual(result['metrics']['/aic_time(us)']['mean'], 4)
        self.assertEqual(result['metrics']['/aiv_time(us)']['mean'], 4)
        self.assertEqual(result['metrics']['/aiv_time(us)']['min_locations'], ['record:3'])
        self.assertEqual(result['metrics']['/aic_time(us)']['missing_or_nonnumeric_count'], 2)
        self.assertEqual(result['metrics']['/bw(GB~1s)']['mean'], 10)
        self.assertEqual(result['records'][0]['raw']['ratio(%)'], '50%')

    def test_missing_requested_field_is_partial_not_zero(self):
        self.add('trace.json', json.dumps([self.event()]), fields=['/args/missing'])
        report = self.run_summary()
        self.assertEqual(report['status'], 'partial')
        self.assertIsNone(report['sources'][0]['metrics']['/args/missing']['mean'])

    def test_msopprof_trailing_delimiter_is_allowed_only_when_cells_are_empty(self):
        self.add('basic.csv', 'Op Name,Task Duration(us),\ntarget,43.001999,\n',
                 fmt='csv', select={'/Op Name': 'target'}, fields=['/Task Duration(us)'])
        report = self.run_summary()
        self.assertEqual(report['status'], 'parsed')
        self.assertEqual(report['sources'][0]['trailing_empty_columns_ignored'], 1)
        self.manifest['sources'] = []
        self.add('bad.csv', 'Op Name,Task Duration(us),\ntarget,43,unlabelled\n',
                 fmt='csv', select={'/Op Name': 'target'})
        self.assertEqual(self.run_summary()['status'], 'partial')

    def test_large_native_timestamp_preserves_fractional_duration(self):
        event = self.event(start='1794195738519907.699', duration=0.013)
        self.add('trace.json', json.dumps([event]))
        result = self.run_summary()['sources'][0]['tracks'][0]
        self.assertEqual(result['union_duration'], .013)
        self.assertEqual(result['span'], .013)
        self.assertEqual(result['interval_union'], [['1794195738519907.699', '1794195738519907.712']])

    def test_intervals_merge_overlap_but_do_not_sum_other_tracks_or_passes(self):
        events = [self.event(0, 10), self.event(5, 10), self.event(20, 5),
                  self.event(0, 25, name='other_pipe')]
        self.add('timeline.json', json.dumps(events), role='timeline', select={})
        self.add('replay.json', json.dumps([self.event(0, 100)]), pass_id='p2')
        report = self.run_summary()
        tracks = report['sources'][0]['tracks']
        self.assertEqual(tracks[0]['union_duration'], 20)
        self.assertEqual(tracks[0]['span'], 25)
        self.assertEqual(tracks[0]['uncovered_within_span'], 5)
        self.assertEqual(tracks[1]['union_duration'], 25)
        self.assertEqual(report['sources'][1]['tracks'][0]['union_duration'], 100)
        self.assertNotIn('latency', report)

    def test_input_drift_retains_other_sources_and_never_parses_changed_file(self):
        self.add('old.json', json.dumps([self.event()]))
        self.add('good.json', json.dumps([self.event()]), pass_id='p2')
        (self.root / 'old.json').write_text('[]')
        report = self.run_summary()
        self.assertEqual(report['status'], 'partial')
        self.assertIn('SHA256 mismatch', report['sources'][0]['error'])
        self.assertEqual(report['sources'][0]['records'], [])
        self.assertEqual(report['sources'][1]['status'], 'parsed')

    def test_invalid_duration_unsupported_pair_and_empty_match_are_not_success(self):
        for events, role, select in [
            ([self.event(duration=-1)], 'kernel', {'/name': 'target'}),
            ([dict(name='target', ph='B', ts=0)], 'timeline', {}),
            ([self.event(name='helper')], 'kernel', {'/name': 'target'}),
            ([self.event(start=float('nan'))], 'kernel', {'/name': 'target'}),
        ]:
            self.manifest['sources'] = []
            self.add('trace.json', json.dumps(events), role=role, select=select)
            self.assertEqual(self.run_summary()['status'], 'partial')

    def test_csv_kernel_selection_and_invalid_columns(self):
        source = self.add('basic.csv', 'Op Name,Stream Id\ntarget,1\ntarget,2\n',
                          fmt='csv', select={'/Op Name': 'target'})
        self.assertEqual(self.run_summary()['status'], 'partial')
        source['select']['/Stream Id'] = '2'
        self.assertEqual(self.run_summary()['status'], 'parsed')
        for content in ('a,a\n1,2\n', 'a,b\n1\n', 'a,b\n1,2,3\n'):
            self.manifest['sources'] = []
            self.add('bad.csv', content, fmt='csv', role='metrics', select={})
            self.assertEqual(self.run_summary()['status'], 'partial')

    def test_cli_never_overwrites_and_returns_failure_for_partial(self):
        self.add('trace.json', '[]')
        self.run_summary()
        output = self.root / 'summary.json'
        args = ['--input', str(self.root / 'input.json'), '--output', str(output)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(summary.main(args), 2)
            previous = output.read_bytes()
            self.assertEqual(summary.main(args), 2)
        self.assertEqual(output.read_bytes(), previous)

    def test_source_path_rejects_parent_escape_and_symlink(self):
        outside = self.root.parent / 'outside.json'
        outside.write_text(json.dumps([self.event()]))
        self.manifest['sources'] = [{'id': 'escape', 'path': '../outside.json', 'format': 'trace',
                                     'pass_id': 'p1', 'tool': 'msprof', 'scope_evidence': 'e1',
                                     'role': 'timeline', 'select': {'/name': 'custom_kernel'}}]
        report = self.run_summary()
        self.assertEqual(report['status'], 'partial')
        self.assertIn('relative to manifest directory', report['sources'][0]['error'])
        target = self.root / 'real.json'
        target.write_text(json.dumps([self.event()]))
        link = self.root / 'link.json'
        link.symlink_to(target)
        self.manifest['sources'][0]['path'] = 'link.json'
        report = self.run_summary()
        self.assertEqual(report['status'], 'partial')
        self.assertIn('symlink', report['sources'][0]['error'])


if __name__ == '__main__':
    unittest.main()
