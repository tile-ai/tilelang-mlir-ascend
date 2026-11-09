"""New-contract evidence, safe rejection and control recovery; CPU fixtures only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_search_tools as fixtures
search = fixtures.search


class ExecutionContractTest(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SearchToolsTest()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.store = self.f.store
        self.prefix = '.workspace/example/runs/d/'

    def write_json(self, name, value):
        path = self.f.write(self.prefix + name, json.dumps(value))
        return self.store.reference(path)

    def setup_run(self, mode='implement-and-validate', performance_kind='absolute', diagnostics=False):
        inputs = {key: self.store.reference(self.f.write(self.prefix + key + '.txt', 'synthetic ' + key))
                  for key in ('requirements', 'reference', 'comparator', 'cases', 'protocol', 'environment')}
        self.plan = {'schema': 'operator-harness/evaluation-plan@1', 'project_id': 'example', 'run_id': 'd',
                     'performance_kind': performance_kind, 'statistic': 'mean', 'inputs': inputs,
                     'cases': [{'id': 'small', 'outputs': ['out', 'count'], 'min_samples': 2, 'max_latency_ms': 2.0}]}
        self.plan_ref = self.write_json('plan.json', self.plan)
        self.artifacts = {key: self.store.reference(self.f.write(self.prefix + key + '.bin', 'synthetic ' + key))
                          for key in ('source', 'build', 'launch')}
        candidate = self.f.write(self.prefix + 'CANDIDATE.md',
            '- contract: operator-harness/candidate@2\n- project_id: example\n- candidate_id: C1\n'
            '- search_node_id: N0\n- parent_candidate: not_applicable\n' + ''.join(
                f'- {key}_sha256: {ref["sha256"]}\n' for key, ref in self.artifacts.items()))
        self.f.apply('candidate', id='C1', search_node_id='N0', parent_candidate=None, manifest=candidate)
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
        self.f.apply('run', id='d', search_node_id='N0', stage='develop', candidate_id=None if mode == 'initial-and-validate' else 'C1',
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
        a = self.write_json('accuracy.json', self.accuracy)
        p = self.write_json('performance.json', self.performance)
        self.report['cases'] = [{'id': 'small', 'accuracy': a, 'performance': p}]
        self.report['receipts'] = [self.write_json('receipt.json', self.receipt)]
        validation = self.write_json('validation.json', self.report)
        if self.plan['performance_kind'] == 'none':
            performance = 'not_applicable'
        regression = self.f.write(self.prefix + 'regression.md', 'Synthetic idea diagnosis, not device evidence.')
        self.result = self.f.write(self.prefix + 'RESULT.md',
            '- contract: operator-harness/node-result@4\n- project_id: example\n- run_id: d\n'
            '- node: develop\n- search_node_id: N0\n- status: completed\n' +
            f'- accuracy_status: {accuracy}\n- performance_status: {performance}\n- regression_cause: {cause}\n'
            f'- accuracy_evidence: {validation["path"]}\n- performance_evidence: {validation["path"]}\n'
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
        before = self.store.path.read_bytes()
        self.store.apply(op)
        self.assertEqual(before, self.store.path.read_bytes())
        return op

    def test_valid_case_accepts_and_promotes_exact_candidate(self):
        self.setup_run()
        self.accept()
        self.f.apply('promote-candidate', run_id='d', reason='All frozen cases pass')
        self.assertEqual(self.store.reconcile()['best']['candidate_id'], 'C1')
        before = self.store.path.read_bytes()
        self.store.reconcile()
        self.assertEqual(before, self.store.path.read_bytes())

    def test_wrong_candidate_or_output_cannot_pass_declaration(self):
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
        del self.accuracy['outputs']['count']
        self.refresh()
        with self.assertRaisesRegex(search.Invalid, 'output coverage'):
            self.accept()

    def test_raw_dependency_drift_is_detected_after_acceptance(self):
        self.setup_run()
        self.accept()
        self.f.write(self.prefix + 'raw.bin', 'changed output')
        with self.assertRaisesRegex(search.Invalid, 'artifact changed'):
            self.store.load()

    def test_declared_artifact_index_freezes_transitive_members(self):
        self.setup_run()
        child = self.store.reference(self.f.write(self.prefix + 'nested-output.bin', 'original'))
        index = self.write_json('raw-index.json', {'schema': 'operator-harness/artifact-index@1', 'files': [child]})
        self.accuracy['raw_files'] = [index]
        self.refresh()
        self.accept()
        self.f.write(child['path'], 'changed')
        with self.assertRaisesRegex(search.Invalid, 'artifact changed'):
            self.store.load()

    def test_installed_cli_consumes_new_contract_without_source_checkout(self):
        import subprocess
        self.setup_run()
        installed = subprocess.run([sys.executable, str(fixtures.ROOT / 'scripts/install_codex_skills.py'),
                                    '--workspace', str(self.f.root)], capture_output=True, text=True)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        operation = self.write_json('accept.json', {'event_id': 'installed-accept', 'op': 'accept', 'id': 'd',
                                                   'result': self.result})
        cli = self.f.root / '.agents/skills/harness-resume/scripts/searchctl.py'
        result = subprocess.run([sys.executable, str(cli), '--workspace', str(self.f.root), '--project', 'example',
                                 '--owner', 'session', 'apply', str(self.store.local(operation['path']))],
                                capture_output=True, text=True, cwd=self.f.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.store.load()['runs']['d']['acceptance']['validation']['eligible'])

    def test_plan_drift_rejected_but_can_close_and_replace_task(self):
        self.setup_run(mode='verify-only')
        self.f.write(self.prefix + 'protocol.txt', 'changed evaluator')
        with self.assertRaisesRegex(search.Invalid, 'artifact changed'):
            self.accept()
        self.close()
        self.f.record_run('repair', 'develop', finish=False)
        self.assertIn('repair', self.store.reconcile()['outstanding_runs'])
        self.assertIsNone(self.store.load()['runs']['d']['result'])
        with self.assertRaises(search.Invalid):
            self.accept()

    def test_replacement_and_batch_failure_do_not_erase_sibling(self):
        operation = self.f.batch_fixture()
        self.store.apply(operation)
        self.f.finish_member('d1')
        self.close('d0')
        self.f.apply('select', id='N0', reason='failed member safely discarded')
        self.f.record_run('repair', 'develop', finish=False)
        self.assertTrue(self.store.load()['runs']['d1']['result'])

    def test_new_batch_rejects_different_comparison_before_registering_any_member(self):
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
        before = self.store.path.read_bytes()
        with self.assertRaisesRegex(search.Invalid, 'comparison plans differ'):
            self.store.apply(operation)
        self.assertEqual(before, self.store.path.read_bytes())
        plan['cases'][0]['max_latency_ms'] = 2.0
        self.store.local(ref['path']).write_text(json.dumps(plan))
        self.store.apply(operation)
        self.assertEqual(self.store.load()['batches']['B1']['members'], ['d0', 'd1'])

    def test_unknown_or_duplicate_contract_rejected_even_in_legacy_mode(self):
        for value in ('operator-harness/node-task@999',
                      'operator-harness/node-task@3\n- contract: operator-harness/node-task@3'):
            task = self.f.write(self.prefix + 'TASK.md', f'- contract: {value}\n'
                '- project_id: example\n- run_id: d\n- node: analysis\n- search_node_id: N0\n')
            with self.assertRaisesRegex(search.Invalid, 'contract'):
                self.f.apply('run', id='d', search_node_id='N0', stage='analysis',
                             candidate_id=None, task=task, legacy=True)

    def test_unversioned_new_run_requires_explicit_legacy_import(self):
        task = self.f.write(self.prefix + 'TASK.md',
            '- project_id: example\n- run_id: d\n- node: analysis\n- search_node_id: N0\n')
        with self.assertRaisesRegex(search.Invalid, 'contract'):
            self.f.apply('run', id='d', search_node_id='N0', stage='analysis', candidate_id=None, task=task, legacy=False)

    def test_evaluator_change_blocks_new_acceptance_not_historical_read(self):
        self.setup_run()
        with patch.object(search.evidence, 'evaluator_identity', return_value={'changed': 'tool'}):
            with self.assertRaisesRegex(search.Invalid, 'evaluator changed'):
                self.accept()
        self.accept()
        with patch.object(search.evidence, 'evaluator_identity', return_value={'changed': 'tool'}):
            self.assertTrue(self.store.load()['runs']['d']['result'])

    def test_samples_and_receipt_are_checked_not_only_report_status(self):
        self.setup_run()
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

    def test_valid_slow_idea_can_be_received_but_cannot_be_best(self):
        self.setup_run()
        self.performance['samples_ms'] = [3.0, 4.0]
        self.refresh(performance='failed', cause='idea')
        self.accept()
        with self.assertRaisesRegex(search.Invalid, 'passing evidence'):
            self.f.apply('promote-candidate', run_id='d', reason='must not promote slow result')

    def test_code_only_accepts_code_without_new_validation_or_promotion(self):
        self.setup_run(mode='code-only', performance_kind='none')
        with self.assertRaisesRegex(search.Invalid, 'cannot declare new'):
            self.accept()
        self.refresh(accuracy='not_run')
        self.accept()
        with self.assertRaisesRegex(search.Invalid, 'passing evidence'):
            self.f.apply('promote-candidate', run_id='d', reason='code exists')

    def test_collect_only_requires_matching_diagnostic_artifacts(self):
        self.setup_run(mode='collect-only', performance_kind='none', diagnostics=True)
        self.refresh(accuracy='not_run')
        diagnostic = {'schema': 'operator-harness/diagnostic-evidence@1', 'project_id': 'example',
                      'run_id': 'd', 'candidate': self.candidate_ref, 'artifacts': []}
        ref = self.write_json('diagnostic.json', diagnostic)
        result = self.store.local(self.result)
        result.write_text(result.read_text() + f'- diagnostic_evidence: {ref["path"]}\n')
        with self.assertRaisesRegex(search.Invalid, 'empty diagnostic'):
            self.accept()
        diagnostic['artifacts'] = [self.accuracy['raw_files'][0]]
        self.store.local(ref['path']).write_text(json.dumps(diagnostic))
        self.accept()
        self.assertFalse(self.store.load()['runs']['d']['acceptance']['validation']['eligible'])

    def test_verify_only_cannot_substitute_same_id_manifest(self):
        self.setup_run(mode='verify-only')
        clone = self.f.write(self.prefix + 'OTHER.md', self.store.local(self.candidate_ref['path']).read_text())
        self.candidate_ref = self.store.reference(clone)
        self.report['candidate'] = self.candidate_ref
        self.refresh()
        with self.assertRaisesRegex(search.Invalid, 'read-only validation changed candidate'):
            self.accept()

    def test_initial_path_needs_no_performance_source_but_keeps_case_gate(self):
        self.setup_run(mode='initial-and-validate')
        self.performance['samples_ms'] = [4.0, 5.0]
        self.refresh(performance='failed')
        self.accept()
        self.assertFalse(self.store.load()['runs']['d']['acceptance']['validation']['eligible'])

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

    def control_init(self, steps=1):
        authority = self.f.write(self.prefix + 'authority.md', 'Synthetic bounded authorization')
        self.f.apply('control-init', step_limit=steps, budgets={'device': {'unit': 'attempts', 'limit': 2}},
                     authority=authority)
        return authority

    def test_unknown_job_keeps_budget_and_blocks_closure(self):
        self.control_init()
        self.setup_run()
        self.f.apply('control-reserve', id='R1', run_id='d', pool='device', amount=1)
        def observe(status):
            return self.write_json(status + '.json', {'schema': 'operator-harness/job-observation@1',
                'project_id': 'example', 'run_id': 'd', 'request_id': 'J1', 'status': status,
                'observed_at': '2026-01-01T00:00:00+00:00', 'source': 'synthetic CPU fixture'})['path']
        observation = observe('intent')
        self.f.apply('control-job', id='J1', reservation_id='R1', status='intent', evidence=observation)
        self.f.apply('control-job', id='J1', status='unknown', evidence=observe('unknown'))
        with self.assertRaisesRegex(search.Invalid, 'job'):
            self.close()
        with self.assertRaisesRegex(search.Invalid, 'job'):
            self.f.apply('control-settle', id='R1', used=1, evidence=observation)
        self.f.apply('control-job', id='J1', status='exited', evidence=observe('exited'))
        op = dict(event_id='settle-stable', op='control-settle', id='R1', used=1, evidence=observation)
        self.store.apply(op)
        self.store.apply(op)
        self.accept()
        snapshot = self.store.reconcile()['control']
        self.assertEqual(snapshot['available']['device'], 1)
        self.assertEqual(snapshot['steps_used'], 1)
        with self.assertRaisesRegex(search.Invalid, 'step budget'):
            self.f.record_run('second', finish=False)
        self.assertNotIn('second', self.store.load()['runs'])

    def test_pause_survives_restart_and_extension_does_not_reset_steps(self):
        authority = self.control_init()
        self.setup_run()
        self.f.apply('control-pause', authority=authority)
        self.store = search.Store(self.f.root, 'example', 'session')
        with self.assertRaisesRegex(search.Invalid, 'paused'):
            self.f.apply('control-reserve', id='R', run_id='d', pool='device', amount=1)
        self.accept()  # Existing results can still be reconciled while paused.
        self.f.apply('control-extend', step_limit=2, limits={'device': 3}, authority=authority)
        self.assertTrue(self.store.reconcile()['control']['paused'])
        self.f.apply('control-resume', authority=authority)
        self.f.record_run('second', finish=False)
        self.assertEqual(self.store.reconcile()['control']['steps_used'], 2)


if __name__ == '__main__':
    unittest.main()
