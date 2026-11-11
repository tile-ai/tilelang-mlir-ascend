"""Real CPU processes exercise receipts, timeout and no-duplicate recovery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'assets/skills/harness-develop/scripts/runctl.py'
spec = importlib.util.spec_from_file_location('runctl', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


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

    def test_input_drift_rejects_before_request_creation(self):
        request = self.request('pass')
        self.worker.write_text('changed')
        with self.assertRaisesRegex(runner.projectctl.Invalid, 'input changed'):
            runner.execute(self.workspace, 'example', 'session', request)
        self.assertFalse((self.project / 'runs/d/jobs/job').exists())


if __name__ == '__main__':
    unittest.main()
