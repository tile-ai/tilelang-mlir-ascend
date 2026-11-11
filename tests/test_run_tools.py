"""Real CPU processes exercise receipts, timeout and no-duplicate recovery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'assets/skills/harness-develop/scripts/runctl.py'
spec = importlib.util.spec_from_file_location('runctl', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def execute_request(root, request):
    return runner.execute(runner.projectctl.Workspace(root), 'example', 'session', request)


class RunToolsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = runner.projectctl.Workspace(self.root)
        self.workspace.create('example', 'Example', 'session')
        self.project = self.root / '.workspace/example'
        self.worker = self.project / 'worker.py'

    def request(self, source):
        self.worker.write_text(source)
        return {'schema': 'operator-harness/process-request@1', 'project_id': 'example',
                'run_id': 'd', 'request_id': 'job', 'cwd': '.', 'timeout_seconds': 2,
                'argv': [sys.executable, str(self.worker)],
                'inputs': [{'path': 'worker.py', 'sha256': runner.projectctl.sha(self.worker)}]}

    def control(self, runs=('d',), limit=1, unit='executions'):
        self.store = runner.searchctl.Store(self.root, 'example', 'session')
        self.store.init('fixture', 'fixture-contract', 'fixture', 'fixture')
        authority = self.project / 'authority.md'
        authority.write_text('Fixture authorization')
        self.store.apply({'op': 'control-init', 'event_id': 'control-init', 'step_limit': len(runs),
                          'budgets': {'cpu': {'unit': unit, 'limit': limit,
                                              'exclusive_resource': 'fixture-device'}},
                          'authority': str(authority.relative_to(self.root))})
        for run in runs:
            task = self.project / f'runs/{run}/TASK.md'
            task.parent.mkdir(parents=True)
            task.write_text(f'- project_id: example\n- run_id: {run}\n- node: analysis\n- search_node_id: N0\n')
            self.store.apply({'op': 'run', 'event_id': 'run-' + run, 'id': run, 'stage': 'analysis',
                              'search_node_id': 'N0', 'candidate_id': None, 'legacy': True,
                              'task': str(task.relative_to(self.root))})

    def intent(self, request):
        directory = runner.location(self.workspace, 'example', request['run_id'], request['request_id'])
        directory.mkdir(parents=True)
        runner.projectctl.atomic(directory / 'request.json', request)
        runner.ledger(self.store, request, directory, 'intent')
        return directory

    def test_exit_is_propagated_and_retry_reuses_receipt(self):
        request = self.request("from pathlib import Path; Path('counter').open('a').write('x'); raise SystemExit(7)")
        first = runner.execute(self.workspace, 'example', 'session', request)
        second = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(first, second)
        self.assertEqual(first['exit_codes']['process'], 7)
        self.assertEqual((self.project / 'counter').read_text(), 'x')
        path = self.root / 'request.json'
        path.write_text(json.dumps(request))
        proc = subprocess.run([sys.executable, str(SCRIPT), '--workspace', str(self.root), '--project',
                               'example', '--owner', 'session', 'run', str(path)], capture_output=True)
        self.assertEqual(proc.returncode, 7)
        self.assertEqual((self.project / 'counter').read_text(), 'x')

    def test_intent_without_receipt_is_unknown_and_never_retried(self):
        request = self.request("raise AssertionError('must not run')")
        directory = runner.location(self.workspace, 'example', 'd', 'job')
        directory.mkdir(parents=True)
        runner.projectctl.atomic(directory / 'request.json', request)
        result = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(result['status'], 'unknown')
        self.assertFalse((directory / 'stdout.log').exists())

    def test_timeout_keeps_logs_and_reports_no_device_verification(self):
        request = self.request("import time; print('started', flush=True); time.sleep(10)")
        request['timeout_seconds'] = 0.15
        result = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(result['status'], 'timed_out')
        self.assertEqual(result['device_validation'], 'not_evaluated')
        self.assertNotEqual(result['exit_codes']['process'], 0)
        self.assertIn('started', (self.project / 'runs/d/jobs/job/stdout.log').read_text())

    def test_timeout_cleanup_waits_are_bounded_and_unknown_keeps_reservation(self):
        self.control()
        request = self.request('pass')
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        child = Mock(pid=123456)
        child.wait.side_effect = subprocess.TimeoutExpired(request['argv'], request['timeout_seconds'])
        with patch.object(runner.subprocess, 'Popen', return_value=child), \
                patch.object(runner, 'start_identity', return_value='fixture-start'), \
                patch.object(runner.os, 'killpg'):
            result = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual([call.kwargs.get('timeout') for call in child.wait.call_args_list],
                         [request['timeout_seconds'], 5])
        self.assertFalse(result['process_stopped'])
        self.assertIn('cleanup is unconfirmed', result['error'])
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['job']['status'], 'unknown')
        self.assertFalse(control['reservations']['job-job']['settled'])

    def test_input_drift_rejects_before_request_creation(self):
        request = self.request('pass')
        self.worker.write_text('changed')
        with self.assertRaisesRegex(runner.projectctl.Invalid, 'input changed'):
            runner.execute(self.workspace, 'example', 'session', request)
        self.assertFalse((self.project / 'runs/d/jobs/job').exists())

    def test_cross_run_request_conflict_never_starts_or_settles_original(self):
        self.control(('a', 'b'))
        first = self.request("from pathlib import Path; Path('counter').open('a').write('x')")
        first.update(run_id='a', reservation={'pool': 'cpu', 'amount': 1})
        self.intent(first)
        before = self.store.path.read_bytes()
        second = dict(first, run_id='b')
        with patch.object(runner.subprocess, 'Popen', wraps=subprocess.Popen) as popen:
            result = runner.execute(self.workspace, 'example', 'session', second)
            retried = runner.execute(self.workspace, 'example', 'session', second)
        self.assertEqual(result['status'], 'not_submitted')
        self.assertEqual(retried, result)
        self.assertIn('identity/budget conflict', result['error'])
        popen.assert_not_called()
        self.assertFalse((self.project / 'counter').exists())
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_concurrent_request_collision_runs_only_one_child(self):
        self.control(('a', 'b'), limit=2)
        request = self.request("from pathlib import Path; Path('counter').open('a').write('x')")
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        requests = [dict(request, run_id=run) for run in ('a', 'b')]
        with ProcessPoolExecutor(2) as pool:
            results = list(pool.map(execute_request, [str(self.root)] * 2, requests))
        self.assertEqual(sorted(r['status'] for r in results), ['exited', 'not_submitted'])
        self.assertEqual((self.project / 'counter').read_text(), 'x')
        tree = self.store.load()
        self.assertEqual(len(tree['control']['reservations']), 1)
        self.assertEqual(tree['control']['reservations']['job-job']['used'], 1)

    def test_existing_reservation_requires_exact_budget_and_request(self):
        self.control()
        request = self.request('pass')
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        directory = self.intent(request)
        before = self.store.path.read_bytes()
        for changed in (dict(request, timeout_seconds=3),
                        dict(request, reservation={'pool': 'cpu', 'amount': 0.5})):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'conflict'):
                runner.ledger(self.store, changed, directory, 'intent')
            self.assertEqual(self.store.path.read_bytes(), before)

    def test_intent_transaction_does_not_leave_reservation_after_job_failure(self):
        self.control()
        request = self.request('pass')
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        mutate = runner.searchctl.Store.mutate
        def fail_job(store, tree, operation):
            if operation['op'] == 'control-job':
                raise ValueError('injected job write failure')
            return mutate(store, tree, operation)
        with patch.object(runner.searchctl.Store, 'mutate', fail_job), \
                patch.object(runner.subprocess, 'Popen', wraps=subprocess.Popen) as popen:
            result = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(result['status'], 'not_submitted')
        popen.assert_not_called()
        self.assertFalse(self.store.load()['control']['reservations'])

    def test_started_process_ledger_failure_is_cancelled_and_charged(self):
        self.control()
        request = self.request("import time; time.sleep(10)")
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        ledger = runner.ledger
        def fail_running(store, spec, directory, state, receipt=None):
            if state == 'running':
                raise OSError('injected running observation failure')
            return ledger(store, spec, directory, state, receipt)
        with patch.object(runner, 'ledger', side_effect=fail_running):
            result = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(result['status'], 'interrupted')
        self.assertTrue(result['process_started'])
        self.assertTrue(result['process_stopped'])
        self.assertIsNotNone(result['exit_codes']['process'])
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['job']['status'], 'cancelled')
        self.assertTrue(control['reservations']['job-job']['settled'])
        self.assertEqual(control['reservations']['job-job']['used'], 1)

    def test_unconfirmed_cleanup_keeps_unknown_job_reserved(self):
        self.control()
        request = self.request('pass')
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        directory = self.intent(request)
        receipt = {'schema': 'operator-harness/process-receipt@1', 'project_id': 'example',
                   'run_id': 'd', 'request_id': 'job', 'status': 'interrupted',
                   'request_sha256': self.store.load()['control']['jobs']['job']['request_sha256'],
                   'process_started': True, 'process_stopped': False,
                   'exit_codes': {'process': None}, 'elapsed_seconds': 1}
        runner.projectctl.atomic(directory / 'process-receipt.json', receipt)
        self.assertEqual(runner.control_state(receipt, directory), 'unknown')
        runner.ledger(self.store, request, directory, 'unknown', receipt)
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['job']['status'], 'unknown')
        self.assertFalse(control['reservations']['job-job']['settled'])

    def test_cleanup_failure_recovers_original_process_without_resubmission(self):
        self.control()
        request = self.request('import time; time.sleep(30)')
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        popen, ledger = subprocess.Popen, runner.ledger
        children = []
        def start(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            child.wait = lambda timeout=None: (_ for _ in ()).throw(subprocess.TimeoutExpired(args[0], timeout))
            return child
        def fail_running(store, spec, directory, state, receipt=None):
            if state == 'running':
                raise OSError('injected observation failure')
            return ledger(store, spec, directory, state, receipt)
        try:
            with patch.object(runner.subprocess, 'Popen', side_effect=start), \
                    patch.object(runner.os, 'killpg', side_effect=PermissionError('injected cleanup failure')), \
                    patch.object(runner, 'ledger', side_effect=fail_running):
                result = runner.execute(self.workspace, 'example', 'session', request)
            self.assertEqual(result['status'], 'interrupted')
            self.assertFalse(result['process_stopped'])
            control = self.store.load()['control']
            self.assertEqual(control['jobs']['job']['status'], 'unknown')
            self.assertFalse(control['reservations']['job-job']['settled'])
        finally:
            for child in children:
                del child.wait
                child.kill()
                child.wait(timeout=5)
        with patch.object(runner.subprocess, 'Popen') as launch:
            recovered = runner.execute(self.workspace, 'example', 'session', request)
        launch.assert_not_called()
        self.assertEqual(recovered, result)
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['job']['status'], 'cancelled')
        self.assertTrue(control['reservations']['job-job']['settled'])

    def test_seconds_recovery_requires_usage_for_unobserved_duration(self):
        self.control(unit='seconds')
        request = self.request('pass')
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        directory = self.intent(request)
        receipt = {'schema': 'operator-harness/process-receipt@1', 'project_id': 'example',
                   'run_id': 'd', 'request_id': 'job', 'status': 'interrupted',
                   'request_sha256': self.store.load()['control']['jobs']['job']['request_sha256'],
                   'process_started': True, 'process_stopped': False,
                   'exit_codes': {'process': None}, 'elapsed_seconds': 0.01}
        runner.projectctl.atomic(directory / 'process-receipt.json', receipt)
        runner.ledger(self.store, request, directory, 'unknown', receipt)
        runner.ledger(self.store, request, directory, 'cancelled', receipt)
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['job']['status'], 'cancelled')
        self.assertFalse(control['reservations']['job-job']['settled'])
        self.assertIn('host usage evidence', control['reservations']['job-job']['settlement_required'])

    def test_deferred_settlement_retains_resource_until_explicit_release(self):
        self.control()
        request = self.request("from pathlib import Path; Path('counter').open('a').write('x')")
        request.update(reservation={'pool': 'cpu', 'amount': 1}, defer_settlement=True)
        result = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(result['exit_codes']['process'], 0)
        self.assertEqual(runner.execute(self.workspace, 'example', 'session', request), result)
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['job']['status'], 'exited')
        self.assertFalse(control['reservations']['job-job']['settled'])
        self.assertIn('final resource release', control['reservations']['job-job']['settlement_required'])
        with self.assertRaisesRegex(ValueError, 'budget exhausted|physical resource'):
            self.store.apply({'event_id': 'new-reserve', 'op': 'control-reserve', 'id': 'another',
                              'run_id': 'd', 'pool': 'cpu', 'amount': 1})
        evidence = self.project / 'release.json'
        evidence.write_text('{"fixture": "original resource released; one execution"}')
        self.store.apply({'event_id': 'job-device-settle', 'op': 'control-settle', 'id': 'job-job',
                          'used': 1, 'evidence': str(evidence.relative_to(self.root))})
        self.assertEqual(runner.execute(self.workspace, 'example', 'session', request), result)
        self.assertTrue(self.store.load()['control']['reservations']['job-job']['settled'])
        self.assertNotIn('settlement_required', self.store.load()['control']['reservations']['job-job'])
        self.assertEqual((self.project / 'counter').read_text(), 'x')

    def test_deferred_request_that_never_started_can_settle_zero(self):
        self.control()
        request = self.request('pass')
        request.update(reservation={'pool': 'cpu', 'amount': 1}, defer_settlement=True,
                       argv=[str(self.project / 'missing-executable')])
        result = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(result['status'], 'failed_to_start')
        self.assertFalse(result['process_started'])
        reservation = self.store.load()['control']['reservations']['job-job']
        self.assertTrue(reservation['settled'])
        self.assertEqual(reservation['used'], 0)

    def test_freezing_allows_existing_request_recovery_but_no_new_process(self):
        self.control()
        request = self.request("from pathlib import Path; Path('counter').open('a').write('x')")
        request.update(reservation={'pool': 'cpu', 'amount': 1}, defer_settlement=True)
        result = runner.execute(self.workspace, 'example', 'session', request)
        self.workspace.begin_close('example', 'session', 'frozen')
        with patch.object(runner.subprocess, 'Popen') as popen:
            self.assertEqual(runner.execute(self.workspace, 'example', 'session', request), result)
            with self.assertRaisesRegex(ValueError, 'active project'):
                runner.execute(self.workspace, 'example', 'session', dict(request, request_id='new'))
        popen.assert_not_called()
        self.assertFalse((self.project / 'runs/d/jobs/new').exists())
        evidence = self.project / 'release.json'
        evidence.write_text('{"fixture": "release observed"}')
        self.store.apply({'event_id': 'release-before-freeze', 'op': 'control-settle', 'id': 'job-job',
                          'used': 1, 'evidence': str(evidence.relative_to(self.root))})
        self.assertTrue(self.store.load()['control']['reservations']['job-job']['settled'])

    def test_freezing_between_intent_and_launch_prevents_spawn(self):
        self.control()
        request = self.request("raise AssertionError('must not launch')")
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        ledger = runner.ledger
        def freeze_after_intent(store, spec, directory, state, receipt=None):
            ledger(store, spec, directory, state, receipt)
            if state == 'intent':
                self.workspace.begin_close('example', 'session', 'frozen')
        with patch.object(runner, 'ledger', side_effect=freeze_after_intent), \
                patch.object(runner.subprocess, 'Popen') as popen:
            result = runner.execute(self.workspace, 'example', 'session', request)
        popen.assert_not_called()
        self.assertFalse(result['process_started'])
        control = self.store.load()['control']
        self.assertEqual(control['jobs']['job']['status'], 'not_submitted')
        self.assertEqual(control['reservations']['job-job']['used'], 0)

    def test_legacy_terminal_request_recovers_without_fingerprint_fields(self):
        self.control()
        request = self.request("from pathlib import Path; Path('counter').open('a').write('x')")
        request['reservation'] = {'pool': 'cpu', 'amount': 1}
        first = runner.execute(self.workspace, 'example', 'session', request)
        tree = self.store.load()
        for collection in ('reservations', 'jobs'):
            for value in tree['control'][collection].values():
                value.pop('request_sha256')
        runner.searchctl.atomic(self.store.path, tree)
        second = runner.execute(self.workspace, 'example', 'session', request)
        self.assertEqual(first, second)
        self.assertEqual((self.project / 'counter').read_text(), 'x')


if __name__ == '__main__':
    unittest.main()
