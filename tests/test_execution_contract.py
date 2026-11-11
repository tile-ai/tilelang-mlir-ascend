"""End-to-end @4 evaluation pipeline, contract boundaries, and control recovery."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_search_tools as fixtures
search = fixtures.search
sys.path.insert(0, str(fixtures.ROOT / 'assets/skills/harness-develop/scripts'))
import evaluate
import runctl
from records import metadata, render


class ExecutionContractTest(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SearchToolsTest()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.store = self.f.store
        self.prefix = '.workspace/example/runs/d/'

    def file(self, name, text):
        return self.f.write(self.prefix + name, text)

    def write_json(self, name, value):
        path = self.f.write(self.prefix + name, json.dumps(value))
        return self.store.reference(path)

    def setup_pipeline(self, bad=False, controlled=False, stage='develop',
                       performance='absolute', diagnostics=False, verify_diag=True):
        if controlled:
            authority = self.file('authority.md', 'Fixture CPU budget')
            self.f.apply('control-init', step_limit=4,
                         budgets={'cpu': {'unit': 'seconds', 'limit': 10}}, authority=authority)
        source = self.file('candidate.py', 'def run(x):\n    return x * ' + ('3' if bad else '2') + '\n')
        reference = self.file('reference.py', "def run(state):\n    return state['x'] * 2\n")
        comparator = self.file('comparator.py', "def compare(actual, expected, case):\n    return {'out': actual == expected}\n")
        verifier_code = "\ndef verify_diagnostics(capture, case):\n    assert capture['evidence']['sha256']\n" if verify_diag else ""
        adapter = self.file('adapter.py', '''import importlib.util
from pathlib import Path
import emission

def prepare(context, case):
    root = Path(context['workspace_root'])
    source = root / context['artifacts']['source']['path']
    spec = importlib.util.spec_from_file_location('candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (root / '.workspace/example/calls').open('a') as stream:
        stream.write('prepare\\n')
    return {'x': case['x'], 'call': module.run}

def run(state):
    return state['call'](state['x'])

def synchronize(state):
    pass

def diagnostics(state, case, output, context):
    emission.save(output / 'raw.json', {'observed': run(state)})
    return {'evidence': emission.diagnostics(context, output / 'index.json', emission.files(context['workspace_root'], output))}
''' + verifier_code)
        inputs = {key: self.store.reference(self.file(key + '.txt', key))
                  for key in ('requirements', 'cases', 'protocol', 'environment')}
        inputs.update(reference=self.store.reference(reference), comparator=self.store.reference(comparator))
        plan = {'schema': 'operator-harness/evaluation-plan@1', 'project_id': 'example', 'run_id': 'd',
                'inputs': inputs, 'performance_kind': performance, 'statistic': 'mean',
                'cases': [{'id': 'one', 'x': 7, 'outputs': ['out'], 'min_samples': 2, 'max_latency_ms': 100}],
                'execution': {'adapter': self.store.reference(adapter), 'backend': 'cpu', 'timeout_seconds': 5}}
        plan_path = self.file('plan.json', json.dumps(plan))
        fields = {'contract': 'operator-harness/node-task@4', 'project_id': 'example', 'run_id': 'd',
                  'node': stage, 'search_node_id': 'N0',
                  'operation': 'initial-and-validate' if stage == 'develop' else 'port-and-validate',
                  'performance_required': 'false' if performance == 'none' else 'true',
                  'diagnostics_required': str(diagnostics).lower(), 'evaluation_plan': plan_path,
                  'initial_design': self.file('design.md', 'Multiply input by two'), 'aicpu_policy': 'forbidden'}
        task = self.file('TASK.md', render('Task', fields))
        self.f.apply('run', id='d', stage=stage, candidate_id=None, search_node_id='N0', task=task, legacy=False)
        self.manifest = evaluate.candidate(
            self.store, 'd', 'C1', source,
            self.file('build.json', '{}'), self.file('launch.json', '{}'), None)['candidate']
        self.reservation = {'pool': 'cpu', 'amount': 5} if controlled else None

    def execute_pipeline(self, request='cpu-1'):
        return evaluate.evaluate(self.store, 'd', self.manifest, request, self.reservation)

    def setup_run(self, mode='implement-and-validate', performance_kind='absolute', diagnostics=False):
        inputs = {key: self.store.reference(self.f.write(self.prefix + key + '.txt', 'synthetic ' + key))
                  for key in ('requirements', 'reference', 'comparator', 'cases', 'protocol', 'environment')}
        self.plan = {'schema': 'operator-harness/evaluation-plan@1', 'project_id': 'example', 'run_id': 'd',
                     'performance_kind': performance_kind, 'statistic': 'mean', 'inputs': inputs,
                     'cases': [{'id': 'small', 'outputs': ['out', 'count'], 'min_samples': 2, 'max_latency_ms': 2.0}]}
        self.plan_ref = self.write_json('plan.json', self.plan)
        self.artifacts = {key: self.store.reference(self.f.write(self.prefix + key + '.bin', 'synthetic ' + key))
                          for key in ('source', 'build', 'launch')}
        parent = 'C0' if mode == 'implement-and-validate' else None
        if parent:
            self.f.candidate(parent)
        candidate = self.f.write(self.prefix + 'CANDIDATE.md',
            '- contract: operator-harness/candidate@2\n- project_id: example\n- candidate_id: C1\n'
            f'- search_node_id: N0\n- parent_candidate: {parent or "not_applicable"}\n' + ''.join(
                f'- {key}_sha256: {ref["sha256"]}\n' for key, ref in self.artifacts.items()))
        self.f.apply('candidate', id='C1', search_node_id='N0', parent_candidate=parent, manifest=candidate)
        self.candidate_ref = self.store.reference(candidate)
        required = 'false' if performance_kind == 'none' else 'true'
        self.task_text = ('- contract: operator-harness/node-task@4\n- project_id: example\n- run_id: d\n'
                         '- node: develop\n- search_node_id: N0\n' + f'- operation: {mode}\n'
                         f'- performance_required: {required}\n- diagnostics_required: {str(diagnostics).lower()}\n'
                         f'- evaluation_plan: {self.plan_ref["path"]}\n')
        if mode == 'initial-and-validate':
            design = self.f.write(self.prefix + 'design.md', 'Synthetic initial algorithm plan')
            self.task_text += f'- initial_design: {design}\n'
        self.task = self.f.write(self.prefix + 'TASK.md', self.task_text)
        self.f.apply('run', id='d', search_node_id='N0', stage='develop',
                     candidate_id=None if mode == 'initial-and-validate' else parent or 'C1',
                     task=self.task, legacy=False)
        self.receipt = {'schema': 'operator-harness/execution-receipt@1', 'project_id': 'example', 'run_id': 'd',
                        'candidate_id': 'C1', 'request_id': 'job1', 'status': 'exited',
                        'exit_codes': {'worker': 0, 'transport': 0}, 'valid': True,
                        'artifacts': self.artifacts, 'evaluation_plan': self.plan_ref}
        common = {'project_id': 'example', 'run_id': 'd', 'candidate_id': 'C1', 'case_id': 'small',
                  'request_id': 'job1', 'artifacts': self.artifacts, 'evaluation_plan': self.plan_ref,
                  'raw_files': [self.store.reference(self.f.write(self.prefix + 'raw.bin', 'synthetic output'))]}
        self.accuracy = dict(common, schema='operator-harness/accuracy@1', outputs={'out': True, 'count': True})
        self.performance = dict(common, schema='operator-harness/performance@1', samples_ms=[1.0, 1.5])
        self.report = {'schema': 'operator-harness/validation@1', 'project_id': 'example', 'run_id': 'd',
                       'candidate_id': 'C1', 'candidate': self.candidate_ref, 'artifacts': self.artifacts,
                       'evaluation_plan': self.plan_ref, 'cases': [], 'receipts': [], 'diagnostics': []}
        self.analysis = self.f.write(self.prefix + 'ANALYSIS_DATA.md',
            '- project_id: example\n- run_id: d\n- candidate_id: C1\n- status: ready\n')
        self.refresh()

    def refresh(self, accuracy='passed', performance='passed', cause='none'):
        self.report['cases'] = [{'id': 'small', 'accuracy': self.accuracy, 'performance': self.performance}]
        self.report['receipts'] = [self.receipt]
        validation = self.write_json('validation.json', self.report)
        if self.plan['performance_kind'] == 'none':
            performance = 'not_applicable'
        regression = self.f.write(self.prefix + 'regression.md', 'Synthetic idea diagnosis, not device evidence.')
        self.result = self.f.write(self.prefix + 'RESULT.md',
            '- contract: operator-harness/node-result@4\n- project_id: example\n- run_id: d\n'
            '- node: develop\n- search_node_id: N0\n- status: completed\n' +
            f'- accuracy_status: {accuracy}\n- performance_status: {performance}\n- regression_cause: {cause}\n'
            f'- regression_evidence: {regression}\n- analysis_data: {self.analysis}\n'
            f'- candidate_manifest: {self.candidate_ref["path"]}\n- validation_evidence: {validation["path"]}\n')

    def accept(self):
        return self.f.apply('accept', id='d', result=self.result)

    def close(self, key='d', successor='repair'):
        run = self.store.load()['runs'][key]
        checkpoint = self.f.write(f'.workspace/example/runs/{key}/checkpoint.md',
            f'- project_id: example\n- run_id: {key}\n- node: {run["stage"]}\n'
            f'- search_node_id: {run["search_node_id"]}\n- status: blocked\n')
        raw = self.store.reference(self.f.write(self.prefix + 'reconciliation.txt', 'Synthetic stopped-process observation'))
        close = self.write_json('close.json', {'schema': 'operator-harness/run-closure@1',
            'project_id': 'example', 'run_id': key, 'summary': 'Synthetic scope replacement',
            'writers_stopped': True, 'jobs': [], 'unsettled_resources': [], 'evidence': [raw]})
        op = dict(event_id='close-' + key, op='close-run', id=key, disposition='superseded',
                  reason='repair requires a new scope', checkpoint=checkpoint,
                  evidence=close['path'], successor=successor)
        self.store.apply(op)
        return op

    def test_generated_evidence_accepts_promotes_and_preserves_edits_on_retry(self):
        self.setup_pipeline(controlled=True, diagnostics=True)
        output = self.execute_pipeline()
        self.assertEqual(output['status'], 'complete')
        result_file = self.store.local(output['result'])
        result_file.write_text(result_file.read_text() + '\nUser reviewed notes.\n')
        self.assertEqual(self.execute_pipeline(), output)
        self.assertIn('User reviewed notes.', result_file.read_text())
        self.assertEqual((self.store.records / 'calls').read_text(), 'prepare\n')
        with self.assertRaisesRegex(search.Invalid, 'different artifacts'):
            evaluate.candidate(self.store, 'd', 'C1', self.file('other.py', 'x=1\n'),
                               self.file('build.json', '{}'), self.file('launch.json', '{}'), None)
        self.f.apply('accept', id='d', result=output['result'])
        self.f.apply('promote-candidate', run_id='d', reason='CPU fixture passed')
        self.assertEqual(self.store.reconcile()['best']['candidate_id'], 'C1')
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['cpu-1']['status'], 'exited')
        self.assertTrue(control['reservations']['job-cpu-1']['settled'])
        self.assertFalse(self.store.reconcile()['corrupted_artifacts'])

    def test_wrong_candidate_generates_invalid_receipt_without_timing_or_acceptance(self):
        self.setup_pipeline(bad=True)
        output = self.execute_pipeline()
        self.assertEqual(output['status'], 'accuracy_failed')
        report = json.loads(self.store.local(output['validation']).read_text())
        self.assertNotIn('performance', report['cases'][0])
        self.assertFalse(report['receipts'][0]['valid'])
        with self.assertRaises(ValueError):
            self.f.apply('accept', id='d', result=output['result'])
        self.assertIsNone(self.store.load()['runs']['d']['result'])

    def test_unverified_diagnostics_stay_partial_until_reviewed(self):
        self.setup_pipeline(diagnostics=True, verify_diag=False)
        output = self.execute_pipeline()
        with self.assertRaisesRegex(search.Invalid, 'not ready'):
            self.f.apply('accept', id='d', result=output['result'])

    def test_port_and_collect_only_modes_in_pipeline(self):
        self.setup_pipeline(stage='port-baseline', performance='none', controlled=True)
        output = self.execute_pipeline()
        result = self.store.local(output['result'])
        fields = metadata(result.read_text())
        fields['porting_evidence'] = self.file('PORTING.md', render('Porting', {
            'contract': 'operator-harness/porting@1', 'project_id': 'example', 'run_id': 'd',
            'candidate_id': 'C1', 'status': 'complete', 'aicpu_status': 'not_used'}))
        reviewed = self.file('RESULT.md', render('Reviewed port', fields))
        self.f.apply('accept', id='d', result=reviewed)
        self.assertTrue(self.store.load()['runs']['d']['acceptance']['validation']['eligible'])
        first = self.store.load()['runs']['d']
        plan = json.loads(self.store.local(first['evaluation_plan']['path']).read_text())
        plan['run_id'] = 'collect'
        plan_path = self.file('collect-plan.json', json.dumps(plan))
        task = self.file('collect-TASK.md', render('Collect', {
            'contract': 'operator-harness/node-task@4', 'project_id': 'example', 'run_id': 'collect',
            'node': 'develop', 'search_node_id': 'N0', 'operation': 'collect-only',
            'performance_required': 'false', 'diagnostics_required': 'true', 'evaluation_plan': plan_path}))
        self.f.apply('run', id='collect', stage='develop', candidate_id='C1', search_node_id='N0', task=task, legacy=False)
        collect_out = evaluate.evaluate(self.store, 'collect', self.manifest, 'diagnostics', self.reservation)
        self.f.apply('accept', id='collect', result=collect_out['result'])
        self.assertFalse(self.store.load()['runs']['collect']['acceptance']['validation']['eligible'])

    def test_historical_damage_blocks_promotion_not_show_or_new_run(self):
        self.setup_pipeline()
        output = self.execute_pipeline()
        self.f.apply('accept', id='d', result=output['result'])
        (self.store.records / 'runs/d/evaluations/cpu-1/case-0/outputs.json').unlink()
        self.assertTrue(self.store.load()['runs']['d']['result'])
        self.assertTrue(self.store.reconcile()['corrupted_artifacts'])
        self.f.record_run('next-analysis', finish=False)
        with self.assertRaises((ValueError, OSError)):
            self.f.apply('promote-candidate', run_id='d', reason='missing raw must block reuse')

    def test_evaluator_code_vs_document_drift_and_metadata_bounds(self):
        self.setup_run()
        with patch.object(search.evidence, 'evaluator_identity', return_value={'changed': 'tool'}):
            with self.assertRaisesRegex(search.Invalid, 'evaluator changed'):
                self.accept()
        with patch.object(search.evidence, 'bundle_identity', return_value='docs changed'):
            self.accept()
        text = render('Result', {'status': 'completed'}, '## Explanation\n- status: failed\n```\n- status: running\n```')
        self.assertEqual(metadata(text), {'status': 'completed'})
        with self.assertRaises(ValueError):
            metadata(render('Result', {'status': 'completed'}).replace('- status: completed', '- status: completed\n- status: failed'))

    def test_validation_rejects_mismatched_identities_samples_and_receipts(self):
        self.setup_run()
        self.accuracy['candidate_id'] = 'OTHER'
        self.refresh()
        with self.assertRaisesRegex(search.Invalid, 'identity mismatch'):
            self.accept()
        self.accuracy['candidate_id'] = 'C1'
        self.accuracy['outputs']['count'] = False
        self.refresh()
        with self.assertRaisesRegex(search.Invalid, 'disagrees'):
            self.accept()
        self.accuracy['outputs']['count'] = True
        for samples in ([1.0], [float('nan'), 1.0], [0.0, 1.0], [9.0, 9.0]):
            self.performance['samples_ms'] = samples
            self.refresh()
            with self.subTest(samples=samples), self.assertRaises(search.Invalid):
                self.accept()
        self.performance['samples_ms'] = [1.0, 1.5]
        for field, value in (('status', 'unknown'), ('valid', False), ('exit_codes', {'worker': 1})):
            old = deepcopy(self.receipt)
            self.receipt[field] = value
            self.refresh()
            with self.subTest(field=field), self.assertRaises(search.Invalid):
                self.accept()
            self.receipt = old

    def test_slow_idea_and_code_only_accept_without_promotion_eligibility(self):
        self.setup_run()
        self.performance['samples_ms'] = [3.0, 4.0]
        self.refresh(performance='failed', cause='idea')
        self.accept()
        with self.assertRaisesRegex(search.Invalid, 'passing evidence'):
            self.f.apply('promote-candidate', run_id='d', reason='must not promote slow result')

    def test_batch_comparison_and_partial_member_closure(self):
        operation = self.f.batch_fixture()
        operation['legacy'] = False
        inputs = {key: self.store.reference(path) for key, path in operation['inputs'].items()}
        plans = []
        for member in operation['members']:
            plan = {'schema': 'operator-harness/evaluation-plan@1', 'project_id': 'example',
                    'run_id': member['id'], 'performance_kind': 'measured', 'statistic': 'mean',
                    'inputs': {key: inputs[key] for key in ('requirements', 'cases', 'protocol', 'environment')},
                    'cases': [{'id': 'one', 'outputs': ['out'], 'min_samples': 2, 'max_latency_ms': 2.0}]}
            plan['inputs'].update(reference=inputs['baseline'], comparator=inputs['protocol'])
            ref = self.write_json(member['id'] + '-plan.json', plan)
            task = self.store.local(member['task'])
            task.write_text('- contract: operator-harness/node-task@4\n' + task.read_text() +
                            f'- diagnostics_required: false\n- evaluation_plan: {ref["path"]}\n')
            plans.append((ref, plan))
        ref, plan = plans[1]
        plan['cases'][0]['max_latency_ms'] = 9.0
        self.store.local(ref['path']).write_text(json.dumps(plan))
        with self.assertRaisesRegex(search.Invalid, 'comparison plans differ'):
            self.store.apply(operation)
        plan['cases'][0]['max_latency_ms'] = 2.0
        self.store.local(ref['path']).write_text(json.dumps(plan))
        self.store.apply(operation)
        self.close('d0')
        self.assertEqual(self.store.load()['batches']['B1']['members'], ['d0', 'd1'])

    def test_mechanism_revisit_requires_history_and_new_basis(self):
        plan = {'schema': 'operator-harness/search-plan@1', 'project_id': 'example',
                'mechanism_id': 'reuse-loads', 'expected': 'fewer loads', 'falsifier': 'same load count',
                'keep': 'all cases pass and improve', 'discard': 'no benefit',
                'simplification': 'compare without the added cache', 'prior_nodes': [], 'new_evidence': []}
        ref = self.write_json('plan1.json', plan)
        self.f.apply('add-node', id='N1', parent='N0', hypothesis='reuse', scope='large', plan=ref['path'])
        ref2 = self.write_json('plan2.json', plan)
        with self.assertRaisesRegex(search.Invalid, 'omits matching'):
            self.f.apply('add-node', id='N2', parent='N0', hypothesis='same mechanism', scope='large', plan=ref2['path'])
        plan['prior_nodes'] = ['N1']
        self.write_json('plan2.json', plan)
        with self.assertRaisesRegex(search.Invalid, 'new evidence'):
            self.f.apply('add-node', id='N2', parent='N0', hypothesis='same mechanism', scope='large', plan=ref2['path'])
        plan['changed_conditions'] = 'Different input distribution; verify before applying conclusion'
        self.write_json('plan2.json', plan)
        self.f.apply('add-node', id='N2', parent='N0', hypothesis='same mechanism', scope='large', plan=ref2['path'])
        self.assertEqual(self.store.mechanisms()[-1]['mechanism_id'], 'reuse-loads')

    def test_control_pause_unknown_jobs_and_orphan_detection(self):
        self.setup_pipeline(controlled=True)
        self.f.apply('control-pause', authority=self.prefix + 'authority.md')
        paused = self.execute_pipeline()
        self.assertEqual(paused['status'], 'not_submitted')
        self.assertFalse(self.store.load()['control']['reservations'])
        self.f.apply('control-resume', authority=self.prefix + 'authority.md')
        self.f.apply('control-reserve', id='R1', run_id='d', pool='cpu', amount=1)
        def observe(status):
            return self.write_json(status + '.json', {'schema': 'operator-harness/job-observation@1',
                'project_id': 'example', 'run_id': 'd', 'request_id': 'J1', 'status': status,
                'observed_at': '2026-01-01T00:00:00+00:00', 'source': 'synthetic CPU fixture'})['path']
        self.f.apply('control-job', id='J1', reservation_id='R1', status='intent', evidence=observe('intent'))
        self.f.apply('control-job', id='J1', status='unknown', evidence=observe('unknown'))
        with self.assertRaisesRegex(search.Invalid, 'job'):
            self.close()
        directory = self.store.records / 'orphan'
        directory.mkdir()
        search.atomic(directory / 'state.json', {'supervisor_pid': 1, 'supervisor_start': 'old',
                                                 'pid': 2, 'process_start': 'live'})
        with patch.object(runctl, 'start_identity', side_effect=lambda pid: 'live' if pid == 2 else None):
            status = runctl.status(directory)
            self.assertEqual(status['status'], 'running')
            self.assertTrue(status['orphaned'])
            self.assertEqual(runctl.control_state({'status': 'timed_out'}, directory), 'unknown')
        self.assertEqual(runctl.control_state({'status': 'timed_out'}, directory), 'cancelled')
        self.assertEqual(runctl.control_state({'status': 'failed_to_start'}, directory), 'not_submitted')


if __name__ == '__main__':
    unittest.main()
