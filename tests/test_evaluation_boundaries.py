"""CPU-only execution probes for evaluator binding, device finalization and case lifetime."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_execution_contract as fixtures

evaluate, search, render = fixtures.evaluate, fixtures.search, fixtures.render

# Only the hardware profiler boundary is mocked. The frozen adapter, worker process,
# output generation, finalization and acceptance paths are the production code.
MOCK_NPU = '''
import contextlib
import struct
from types import SimpleNamespace
import torch_npu_measure
@contextlib.contextmanager
def profile_fixture(directory):
    directory.mkdir()
    values = [0] * 16
    values[0], values[1], values[2] = 1 << 32, 10, 1 << 31
    values[14], values[15] = 0, 1000
    (directory / 'ffts_profile.slice_0').write_bytes(struct.pack('<16q', *values))
    yield SimpleNamespace(prof_if=SimpleNamespace(prof_path=directory), step=lambda: None)
torch_npu_measure.capture_profile = profile_fixture

def measurement(state, case, output):
    return dict(call=lambda: run(state), output=output, flush=lambda: None,
        synchronize=lambda: None, frequency=lambda: (1000,),
        check_reservation=lambda: None, verify_profile=lambda path, parsed: None,
        protocol='cpu-fixture', expected_frequency=(1000,), flush_bytes=1024,
        warmup=0, repeats=2)
'''

LIFETIMES = '''
import weakref
allocations = []
original_prepare = prepare
original_run = run
class Buffers:
    pass

def prepare(context, case):
    assert not any(ref() is not None for ref in allocations), 'previous case buffers still alive'
    state = original_prepare(context, case)
    state['buffers'] = Buffers()
    allocations.append(weakref.ref(state['buffers']))
    state['root'] = Path(context['workspace_root'])
    state['case'] = case['id']
    return state

def run(state):
    with (state['root'] / '.workspace/example/lifecycle').open('a') as stream:
        stream.write('run:' + state['case'] + '\\n')
    return original_run(state)

def teardown(state):
    with (state['root'] / '.workspace/example/lifecycle').open('a') as stream:
        stream.write('teardown:' + state['case'] + '\\n')
'''


class EvaluationBoundariesTest(unittest.TestCase):
    def setUp(self):
        self.t = fixtures.ExecutionContractTest()
        self.t.setUp()
        self.addCleanup(self.t.doCleanups)

    def device_receipt(self, pending, name='device.json', **changes):
        t = self.t
        context = json.loads(t.store.local(pending['context']).read_text())
        receipt = {key: context[key] for key in ('project_id', 'run_id', 'candidate_id', 'request_id',
                                                 'artifacts', 'evaluation_plan')}
        receipt.update(schema='operator-harness/device-receipt@1', status='released',
                       worker_result=pending['worker_result'], exit_codes={'guard': 0, 'transport': 0}, conflicts=[],
                       raw_files=[t.store.reference(t.file('guard.log', 'CPU fixture: released without conflict'))])
        receipt.update(changes)
        return t.file(name, json.dumps(receipt))

    def test_new_plan_cannot_omit_or_empty_execution(self):
        original = self.t.file
        def omit(name, text):
            if name == 'plan.json':
                data = json.loads(text)
                data.pop('execution')
                text = json.dumps(data)
            return original(name, text)
        self.t.file = omit
        with self.assertRaisesRegex(search.Invalid, 'frozen execution adapter'):
            self.t.setup_pipeline()
        self.assertNotIn('d', self.t.store.load()['runs'])
        path = self.t.prefix + 'plan.json'
        data = json.loads(self.t.store.local(path).read_text())
        for execution in (None, {}, False, []):
            data['execution'] = execution
            self.t.store.local(path).write_text(json.dumps(data))
            with self.subTest(execution=execution), self.assertRaisesRegex(search.Invalid, 'frozen execution adapter'):
                search.evidence.plan(self.t.store, self.t.store.reference(path))

    def test_new_collect_only_requires_frozen_evaluation_plan(self):
        t = self.t
        t.f.candidate('C0')
        task = t.file('collect-task.md', render('Collect', {
            'contract': 'operator-harness/node-task@4', 'project_id': 'example', 'run_id': 'collect',
            'node': 'develop', 'search_node_id': 'N0', 'operation': 'collect-only',
            'performance_required': 'false', 'diagnostics_required': 'true'}))
        with self.assertRaisesRegex(search.Invalid, 'evaluation_plan'):
            t.f.apply('run', id='collect', stage='develop', candidate_id='C0', search_node_id='N0',
                      task=task, legacy=False)
        self.assertNotIn('collect', t.store.load()['runs'])

    def test_pending_device_requires_final_receipt_and_finalizes_without_rerun(self):
        t = self.t
        t.setup_pipeline(backend='torch_npu', adapter_extra=MOCK_NPU)
        pending = t.execute_pipeline()
        self.assertEqual(pending['status'], 'pending_guard_receipt')
        output = t.store.local(pending['context']).parent
        self.assertFalse((output / 'validation.json').exists())
        self.assertFalse((output / 'RESULT.generated.md').exists())
        calls = (t.store.records / 'calls').read_text()
        for fields in ({'status': 'pending'}, {'conflicts': ['another device job']},
                       {'exit_codes': {'guard': 1}}, {'exit_codes': {'transport': 0}},
                       {'raw_files': []}, {'request_id': 'other'},
                       {'worker_result': t.store.reference(t.file('wrong-result.json', '{}'))}):
            path = self.device_receipt(pending, **fields)
            with self.subTest(fields=fields), self.assertRaises(search.Invalid):
                evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1', device_receipt_path=path)
            self.assertFalse((output / 'device-finalization.json').exists())
            self.assertEqual((t.store.records / 'calls').read_text(), calls)
        path = self.device_receipt(pending)
        result = evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1', device_receipt_path=path)
        self.assertEqual(result['status'], 'complete')
        t.store.local(result['result']).write_text(t.store.local(result['result']).read_text() + '\nReviewed notes.\n')
        self.assertEqual(t.execute_pipeline(), result)
        self.assertIn('Reviewed notes.', t.store.local(result['result']).read_text())
        self.assertEqual((t.store.records / 'calls').read_text(), calls)
        t.f.apply('accept', id='d', result=result['result'])
        t.f.apply('promote-candidate', run_id='d', reason='CPU mocked device finalization test')
        self.assertEqual(t.store.load()['best']['candidate_id'], 'C1')
        t.store.local(path).write_text('{}')
        with self.assertRaises(ValueError):
            t.f.apply('promote-candidate', run_id='d', reason='changed device receipt must fail')

    def test_device_accuracy_only_and_collect_only_also_wait_for_guard(self):
        t = self.t
        t.setup_pipeline(backend='torch_npu', adapter_extra=MOCK_NPU, performance='none')
        pending = t.execute_pipeline()
        self.assertEqual(pending['status'], 'pending_guard_receipt')
        result = evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1',
                                   device_receipt_path=self.device_receipt(pending))
        t.f.apply('accept', id='d', result=result['result'])
        run = t.store.load()['runs']['d']
        plan = json.loads(t.store.local(run['evaluation_plan']['path']).read_text())
        plan['run_id'] = 'collect'
        plan_path = t.file('collect-plan.json', json.dumps(plan))
        task = t.file('collect-task.md', render('Collect', {
            'contract': 'operator-harness/node-task@4', 'project_id': 'example', 'run_id': 'collect',
            'node': 'develop', 'search_node_id': 'N0', 'operation': 'collect-only',
            'performance_required': 'false', 'diagnostics_required': 'false', 'evaluation_plan': plan_path}))
        t.f.apply('run', id='collect', stage='develop', candidate_id='C1', search_node_id='N0', task=task, legacy=False)
        pending = evaluate.evaluate(t.store, 'collect', t.manifest, 'collect-job')
        self.assertEqual(pending['status'], 'pending_guard_receipt')
        result = evaluate.evaluate(t.store, 'collect', t.manifest, 'collect-job',
                                   device_receipt_path=self.device_receipt(pending, name='collect-device.json'))
        t.f.apply('accept', id='collect', result=result['result'])
        self.assertFalse(t.store.load()['runs']['collect']['acceptance']['validation']['eligible'])

    def test_controlled_device_keeps_exclusive_reservation_until_final_usage(self):
        t = self.t
        authority = t.file('authority.md', 'Synthetic exclusive device budget')
        t.f.apply('control-init', step_limit=4, budgets={
            'device': {'unit': 'seconds', 'limit': 10, 'exclusive_resource': 'fixture-device'}}, authority=authority)
        t.setup_pipeline(backend='torch_npu', adapter_extra=MOCK_NPU)
        t.reservation = {'pool': 'device', 'amount': 5}
        pending = t.execute_pipeline()
        self.assertEqual(pending['status'], 'pending_guard_receipt')
        original_calls = (t.store.records / 'calls').read_text()
        self.assertFalse(t.store.load()['control']['reservations']['job-cpu-1']['settled'])
        with self.assertRaisesRegex(search.Invalid, 'physical resource'):
            t.f.apply('control-reserve', id='next-device', run_id='d', pool='device', amount=1)
        for usage in (None, {'pool': 'other', 'unit': 'seconds', 'used': 1},
                      {'pool': 'device', 'unit': 'executions', 'used': 1},
                      {'pool': 'device', 'unit': 'seconds', 'used': 0},
                      {'pool': 'device', 'unit': 'seconds', 'used': 0.000001}):
            path = self.device_receipt(pending, resource_usage=usage)
            with self.subTest(usage=usage), self.assertRaisesRegex(search.Invalid, 'resource usage|shorter'):
                evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1', t.reservation, path)
            self.assertFalse(t.store.load()['control']['reservations']['job-cpu-1']['settled'])
        path = self.device_receipt(pending, resource_usage={'pool': 'device', 'unit': 'seconds', 'used': 2.5})
        result = evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1', t.reservation, path)
        reservation = t.store.load()['control']['reservations']['job-cpu-1']
        self.assertTrue(reservation['settled'])
        self.assertEqual(reservation['used'], 2.5)
        self.assertEqual(reservation['evidence'], t.store.reference(path))
        self.assertEqual(t.execute_pipeline(), result)
        self.assertEqual((t.store.records / 'calls').read_text(), original_calls)
        events = t.store.load()['events']
        self.assertEqual(sum(event['operation']['event_id'] == 'cpu-1-device-settle' for event in events), 1)
        t.f.apply('accept', id='d', result=result['result'])
        t.f.record_run('next', finish=False)
        t.f.apply('control-reserve', id='next-device', run_id='next', pool='device', amount=1)

    def test_freezing_finishes_original_device_request_without_starting_new_work(self):
        t = self.t
        t.setup_pipeline(controlled=True, backend='torch_npu', adapter_extra=MOCK_NPU)
        pending = t.execute_pipeline()
        path = self.device_receipt(pending, resource_usage={'pool': 'cpu', 'unit': 'seconds', 'used': 2.5})
        calls = (t.store.records / 'calls').read_text()
        t.store.workspace.begin_close('example', t.store.owner, 'frozen')
        result = evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1', t.reservation, path)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(t.store.load()['control']['reservations']['job-cpu-1']['settled'])
        self.assertEqual(t.execute_pipeline(), result)
        with self.assertRaisesRegex(ValueError, 'active|lifecycle'):
            t.execute_pipeline('new-request')
        self.assertFalse((t.store.records / 'runs/d/jobs/new-request').exists())
        self.assertFalse((t.store.records / 'runs/d/evaluations/new-request/context.json').exists())
        self.assertEqual((t.store.records / 'calls').read_text(), calls)

    def test_controlled_device_request_consumes_exactly_one_execution(self):
        t = self.t
        authority = t.file('authority.md', 'Synthetic execution budget')
        t.f.apply('control-init', step_limit=4,
                  budgets={'device': {'unit': 'executions', 'limit': 5}}, authority=authority)
        t.setup_pipeline(backend='torch_npu', adapter_extra=MOCK_NPU)
        t.reservation = {'pool': 'device', 'amount': 1}
        pending = t.execute_pipeline()
        for used in (0, 2, 1.0):
            path = self.device_receipt(pending, resource_usage={'pool': 'device', 'unit': 'executions', 'used': used})
            with self.subTest(used=used), self.assertRaises(ValueError):
                evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1', t.reservation, path)
        path = self.device_receipt(pending, resource_usage={'pool': 'device', 'unit': 'executions', 'used': 1})
        result = evaluate.evaluate(t.store, 'd', t.manifest, 'cpu-1', t.reservation, path)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(t.store.load()['control']['reservations']['job-cpu-1']['used'], 1)

    def test_backend_metric_change_cannot_replace_existing_best(self):
        t = self.t
        t.setup_pipeline()
        first = t.execute_pipeline()
        # Simulate a historical acceptance with the exact pre-metric hash.
        with patch.object(search.evidence, 'comparison_identity', search.evidence.previous_comparison_identity):
            t.f.apply('accept', id='d', result=first['result'])
        t.f.apply('promote-candidate', run_id='d', reason='CPU wall comparison')
        plan = json.loads(t.store.local(t.store.load()['runs']['d']['evaluation_plan']['path']).read_text())
        cpu_identity = search.evidence.comparison_identity(plan)
        plan['run_id'] = 'npu'
        source = t.store.local(plan['execution']['adapter']['path']).read_text()
        plan['execution'].update(backend='torch_npu', adapter=t.store.reference(t.file('npu-adapter.py', source + MOCK_NPU)))
        self.assertNotEqual(cpu_identity, search.evidence.comparison_identity(plan))
        path = t.file('npu-plan.json', json.dumps(plan))
        task = t.file('npu-task.md', render('Verify', {
            'contract': 'operator-harness/node-task@4', 'project_id': 'example', 'run_id': 'npu',
            'node': 'develop', 'search_node_id': 'N0', 'operation': 'verify-only',
            'performance_required': 'true', 'diagnostics_required': 'false', 'evaluation_plan': path}))
        t.f.apply('run', id='npu', stage='develop', candidate_id='C1', search_node_id='N0', task=task, legacy=False)
        pending = evaluate.evaluate(t.store, 'npu', t.manifest, 'npu-job')
        result = evaluate.evaluate(t.store, 'npu', t.manifest, 'npu-job',
                                   device_receipt_path=self.device_receipt(pending))
        t.f.apply('accept', id='npu', result=result['result'])
        with self.assertRaisesRegex(search.Invalid, 'comparison changed'):
            t.f.apply('promote-candidate', run_id='npu', reason='incomparable metric')
        self.assertEqual(t.store.load()['best']['run_id'], 'd')
        plan['execution']['metric'] = 'cpu_wall_ms'
        path = t.file('wrong-metric.json', json.dumps(plan))
        with self.assertRaisesRegex(search.Invalid, 'metric'):
            search.evidence.plan(t.store, t.store.reference(path))

    def test_legacy_comparison_requires_intact_original_plan_and_preserves_acceptance(self):
        t = self.t
        t.setup_pipeline()
        first = t.execute_pipeline()
        with patch.object(search.evidence, 'comparison_identity', search.evidence.previous_comparison_identity):
            t.f.apply('accept', id='d', result=first['result'])
        t.f.apply('promote-candidate', run_id='d', reason='Historical CPU comparison')
        old_run = t.store.load()['runs']['d']
        old_acceptance = json.loads(json.dumps(old_run['acceptance']))
        plan = json.loads(t.store.local(old_run['evaluation_plan']['path']).read_text())
        plan['run_id'] = 'new'
        path = t.file('new-plan.json', json.dumps(plan))
        task = t.file('new-task.md', render('Verify', {
            'contract': 'operator-harness/node-task@4', 'project_id': 'example', 'run_id': 'new',
            'node': 'develop', 'search_node_id': 'N0', 'operation': 'verify-only',
            'performance_required': 'true', 'diagnostics_required': 'false', 'evaluation_plan': path}))
        t.f.apply('run', id='new', stage='develop', candidate_id='C1', search_node_id='N0', task=task, legacy=False)
        result = evaluate.evaluate(t.store, 'new', t.manifest, 'new-job')
        t.f.apply('accept', id='new', result=result['result'])
        # Damage only an old raw output. The new execution remains valid.
        old_output = t.store.records / 'runs/d/evaluations/cpu-1/case-0/outputs.json'
        original = old_output.read_bytes()
        old_output.write_text('corrupted')
        with self.assertRaises(ValueError):
            t.f.apply('promote-candidate', run_id='new', reason='Damaged old comparison must not migrate')
        old_output.write_bytes(original)
        t.f.apply('promote-candidate', run_id='new', reason='Same frozen CPU metric under current evaluator')
        tree = t.store.load()
        self.assertEqual(tree['best']['run_id'], 'new')
        self.assertEqual(tree['runs']['d']['acceptance'], old_acceptance)
        self.assertNotEqual(tree['best']['comparison'], old_acceptance['validation']['comparison'])

    def test_case_buffers_are_released_before_next_prepare_and_all_accuracy_precedes_timing(self):
        t = self.t
        cases = [{'id': str(i), 'x': i, 'outputs': ['out'], 'min_samples': 2, 'max_latency_ms': 100} for i in range(3)]
        t.setup_pipeline(cases=cases, adapter_extra=LIFETIMES)
        result = t.execute_pipeline()
        self.assertEqual(result['status'], 'complete')
        events = (t.store.records / 'lifecycle').read_text().splitlines()
        self.assertEqual(events[:6], ['run:0', 'teardown:0', 'run:1', 'teardown:1', 'run:2', 'teardown:2'])
        self.assertEqual(events[6:], ['run:0', 'run:0', 'teardown:0', 'run:1', 'run:1', 'teardown:1',
                                      'run:2', 'run:2', 'teardown:2'])
        t.f.apply('accept', id='d', result=result['result'])

    def test_accuracy_failure_releases_each_case_and_skips_timing(self):
        t = self.t
        t.setup_pipeline(bad=True, adapter_extra=LIFETIMES)
        result = t.execute_pipeline()
        self.assertEqual(result['status'], 'accuracy_failed')
        self.assertEqual((t.store.records / 'lifecycle').read_text().splitlines(), ['run:one', 'teardown:one'])

    def test_cleanup_runs_on_execution_exception_and_cleanup_failure_blocks_success(self):
        t = self.t
        t.setup_pipeline(adapter_extra=LIFETIMES + '''
original_teardown = teardown

def run(state):
    raise RuntimeError('execution failure fixture')

def teardown(state):
    original_teardown(state)
    raise RuntimeError('cleanup failure fixture')
''')
        result = t.execute_pipeline()
        self.assertNotEqual(result['process']['exit_codes']['process'], 0)
        self.assertEqual((t.store.records / 'lifecycle').read_text().splitlines(), ['teardown:one'])
        error = (t.store.records / 'runs/d/jobs/cpu-1/stderr.log').read_text()
        self.assertIn('execution failure fixture', error)
        self.assertIn('cleanup failure fixture', error)
        self.assertFalse((t.store.records / 'runs/d/evaluations/cpu-1/validation.json').exists())


if __name__ == '__main__':
    unittest.main()
