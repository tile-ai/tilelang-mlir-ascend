"""Exercise copied skills through their public CLIs in a fresh workspace."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InstalledPipelineTest(unittest.TestCase):
    def test_good_and_bad_candidates_from_fresh_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            installed = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/install_codex_skills.py'), '--workspace', str(workspace)],
                cwd=workspace, capture_output=True, text=True)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            skills = workspace / '.agents/skills'

            for bad in (False, True):
                project = 'bad' if bad else 'good'
                prefix = f'.workspace/{project}/runs/d/'

                def tool(skill, script, *arguments, expected=0):
                    result = subprocess.run(
                        [sys.executable, str(skills / skill / 'scripts' / script),
                         '--workspace', str(workspace), '--project', project, '--owner', 'session',
                         *arguments], cwd=workspace, capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected, result.stderr + result.stdout)
                    return json.loads(result.stdout) if result.stdout else None

                def write(name, content):
                    path = workspace / (prefix + name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                    return str(path.relative_to(workspace))

                def reference(path):
                    return {'path': path, 'sha256': hashlib.sha256((workspace / path).read_bytes()).hexdigest()}

                def apply(event, operation, expected=0, **fields):
                    path = write(event + '.json', json.dumps(dict(event_id=event, op=operation, **fields)))
                    return tool('harness-resume', 'searchctl.py', 'apply', str(workspace / path), expected=expected)

                tool('harness-new-project', 'projectctl.py', 'create', '--title', 'Installed CPU fixture')
                tool('harness-resume', 'searchctl.py', 'init', '--operator', 'double',
                     '--task-contract', 'fixture', '--hypothesis', 'multiply by two', '--scope', 'CPU')
                source = write('candidate.py', f'def run(x):\n    return x * {3 if bad else 2}\n')
                adapter = write('adapter.py', '''import importlib.util
from pathlib import Path

def prepare(context, case):
    root = Path(context['workspace_root'])
    spec = importlib.util.spec_from_file_location('candidate', root / context['artifacts']['source']['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (root / context['output'] / 'preparations.txt').open('a') as stream:
        stream.write('prepared\\n')
    return {'x': case['x'], 'run': module.run}

def run(state):
    return state['run'](state['x'])

def synchronize(state):
    pass
''')
                ref = write('reference.py', "def run(state):\n    return state['x'] * 2\n")
                comparator = write('comparator.py', "def compare(actual, expected, case):\n    return {'out': actual == expected}\n")
                inputs = {key: reference(write(key + '.txt', key))
                          for key in ('requirements', 'cases', 'protocol', 'environment')}
                inputs.update(reference=reference(ref), comparator=reference(comparator))
                plan = write('plan.json', json.dumps({
                    'schema': 'operator-harness/evaluation-plan@1', 'project_id': project, 'run_id': 'd',
                    'inputs': inputs, 'performance_kind': 'absolute', 'statistic': 'mean',
                    'execution': {'backend': 'cpu', 'metric': 'cpu_wall_ms',
                                  'adapter': reference(adapter), 'timeout_seconds': 5},
                    'cases': [{'id': 'one', 'x': 7, 'outputs': ['out'],
                               'min_samples': 2, 'max_latency_ms': 100}]}))
                fields = dict(contract='operator-harness/node-task@4', project_id=project, run_id='d',
                              node='develop', search_node_id='N0', operation='initial-and-validate',
                              performance_required='true', diagnostics_required='false',
                              evaluation_plan=plan, initial_design=write('design.md', 'Multiply by two'))
                task = write('TASK.md', '# Task\n\n<!-- harness-metadata -->\n' + ''.join(
                    f'- {key}: {value}\n' for key, value in fields.items()) + '<!-- /harness-metadata -->\n')
                apply('register', 'run', id='d', stage='develop', search_node_id='N0',
                      candidate_id=None, task=task, legacy=False)
                candidate = tool('harness-develop', 'evaluate.py', 'candidate', '--run', 'd', '--id', 'C1',
                                 '--source', source, '--build', write('build.json', '{}'),
                                 '--launch', write('launch.json', '{}'))['candidate']
                arguments = ('run', '--run', 'd', '--candidate', candidate, '--request', 'd-attempt1')
                output = tool('harness-develop', 'evaluate.py', *arguments, expected=1 if bad else 0)
                preparations = workspace / prefix / 'evaluations/d-attempt1/preparations.txt'
                before = preparations.read_bytes()
                retry = tool('harness-develop', 'evaluate.py', *arguments, expected=1 if bad else 0)
                self.assertEqual(output, retry)
                self.assertEqual(preparations.read_bytes(), before, 'retry re-executed the candidate')
                if bad:
                    apply('accept-bad', 'accept', expected=1, id='d', result=output['result'])
                    worker = json.loads((workspace / prefix / 'evaluations/d-attempt1/worker-result.json').read_text())
                    self.assertTrue(all('performance' not in case for case in worker['cases']))
                    self.assertIsNone(tool('harness-resume', 'searchctl.py', 'show').get('best'))
                else:
                    apply('accept-good', 'accept', id='d', result=output['result'])
                    apply('promote-good', 'promote-candidate', run_id='d', reason='Frozen CPU fixture passed')
                    self.assertEqual(tool('harness-resume', 'searchctl.py', 'show')['best']['candidate_id'], 'C1')


if __name__ == '__main__':
    unittest.main()
