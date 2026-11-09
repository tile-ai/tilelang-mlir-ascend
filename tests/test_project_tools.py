"""CPU-only project isolation, lifecycle, recovery and installed CLI checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'assets/skills/harness-new-project/scripts/projectctl.py'
sys.path.insert(0, str(SCRIPT.parent))
import projectctl as project
sys.path.insert(0, str(ROOT / 'assets/skills/harness-resume/scripts'))
import searchctl as search


class ProjectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = project.Workspace(self.root)
        self.ws.create('first', 'First project', 'session1')

    def checkpoint(self, project_id='first', filename='lifecycle/close.json', **extra):
        path = self.ws.project(project_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(schema='operator-harness/project-checkpoint@1', project_id=project_id,
                    summary='Synthetic fixture, no hardware claims', writers_stopped=True,
                    jobs=[], unsettled_resources=[], open_runs=[], paused_runs={}, reviewed_changes=True)
        data.update(extra)
        path.write_text(json.dumps(data))
        return filename

    def repo(self):
        repo = self.root / 'original'
        repo.mkdir()
        project.git(repo, 'init')
        project.git(repo, 'config', 'user.name', 'Fixture')
        project.git(repo, 'config', 'user.email', 'fixture@example.invalid')
        (repo / 'code.txt').write_text('base\n')
        project.git(repo, 'add', 'code.txt')
        project.git(repo, 'commit', '-m', 'fixture')
        return repo

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), '--workspace', str(self.root), *args],
                              capture_output=True, text=True)

    def test_names_and_metadata_cannot_escape_or_overwrite(self):
        for value in ('../escape', '/tmp/escape', 'environments', 'Upper', 'x/y', ''):
            with self.subTest(value=value), self.assertRaises(project.Invalid):
                self.ws.create(value, 'Title', 'session')
        before = (self.ws.project('first') / 'project.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.ws.create('first', 'Other', 'session')
        self.assertEqual((self.ws.project('first') / 'project.json').read_bytes(), before)
        (self.ws.directory / 'alias').symlink_to(self.ws.project('first'), target_is_directory=True)
        with self.assertRaises(project.Invalid):
            self.ws.load('alias')
        (self.ws.project('first') / 'escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(project.Invalid):
            self.ws.local('first', 'escape/file')

    def test_discovery_has_no_global_current_and_is_read_only(self):
        self.assertEqual(self.ws.resolve(), 'first')
        self.ws.create('second', 'Second', 'session2')
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.ws.directory.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(project.Invalid, 'unique'):
            self.ws.resolve()
        self.assertEqual(len(self.ws.list()), 2)
        self.assertEqual(self.cli('list').returncode, 0)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.ws.directory.rglob('*') if p.is_file()})

    def test_conflicting_session_needs_handoff_evidence(self):
        with self.assertRaises(project.Invalid):
            self.ws.begin_close('first', 'session2', 'frozen')
        with self.assertRaises(project.Invalid):
            self.ws.claim('first', 'session2')
        evidence = self.checkpoint(previous_owner_stopped=False)
        with self.assertRaises(project.Invalid):
            self.ws.claim('first', 'session2', 'session1', evidence)
        evidence = self.checkpoint(previous_owner_stopped=True)
        self.ws.claim('first', 'session2', 'session1', evidence)
        self.assertEqual(self.ws.load('first')['owner'], 'session2')

    def test_two_projects_have_distinct_worktrees_and_keep_source_dirty(self):
        source = self.repo()
        (source / 'code.txt').write_text('original dirty\n')
        self.ws.create('second', 'Second', 'session2')
        self.ws.add_repo('first', 'session1', 'operator', source, 'HEAD')
        self.ws.add_repo('second', 'session2', 'operator', source, 'HEAD')
        first = self.ws.project('first') / 'repos/operator'
        second = self.ws.project('second') / 'repos/operator'
        (first / 'code.txt').write_text('first only\n')
        (first / 'extra.txt').write_text('untracked\n')
        self.assertEqual((second / 'code.txt').read_text(), 'base\n')
        self.assertEqual((source / 'code.txt').read_text(), 'original dirty\n')
        self.assertNotEqual(project.git(first, 'branch', '--show-current'), project.git(second, 'branch', '--show-current'))
        checkpoint = self.checkpoint()
        self.ws.begin_close('first', 'session1', 'archived')
        self.ws.finish_close('first', 'session1', checkpoint)
        inventory = self.ws.load('first')['closed_snapshot']['repositories']['operator']
        self.assertIn('first only', inventory['tracked_patch'])
        self.assertIn('extra.txt', inventory['untracked'])
        self.assertEqual([p['id'] for p in self.ws.list()], ['second'])
        self.assertEqual(len(self.ws.list(True)), 2)
        self.ws.restore('first', 'session1', self.checkpoint(filename='lifecycle/restore.json'))
        self.assertEqual((first / 'code.txt').read_text(), 'first only\n')
        self.assertTrue((first / 'extra.txt').exists())

    def test_interrupted_repo_registration_recovers_without_new_worktree(self):
        source = self.repo()
        original = self.ws.save
        def crash(data, action, detail=None):
            if action == 'repository-ready':
                raise OSError('simulated coordinator interruption')
            return original(data, action, detail)
        with patch.object(self.ws, 'save', side_effect=crash), self.assertRaises(OSError):
            self.ws.add_repo('first', 'session1', 'operator', source, 'HEAD')
        self.assertEqual(self.ws.load('first')['repositories']['operator']['state'], 'preparing')
        before = project.git(source, 'worktree', 'list', '--porcelain')
        self.ws.add_repo('first', 'session1', 'operator', source, 'HEAD')
        self.assertEqual(project.git(source, 'worktree', 'list', '--porcelain'), before)
        self.assertEqual(self.ws.load('first')['repositories']['operator']['state'], 'ready')

    def test_unknown_jobs_keep_freezing_and_block_search_mutation(self):
        store = search.Store(self.root, 'first', 'session1')
        store.init('operator-group', 'contract', 'hypothesis', 'multiple operators')
        self.ws.begin_close('first', 'session1', 'frozen')
        with self.assertRaises(project.Invalid):
            self.ws.finish_close('first', 'session1', self.checkpoint(jobs=[{'id': 'unknown'}]))
        with self.assertRaises(project.Invalid):
            store.apply({'event_id': 'e1', 'op': 'add-node', 'id': 'N1', 'parent': 'N0',
                         'hypothesis': 'other', 'scope': 'all'})
        self.assertEqual(self.ws.load('first')['status'], 'freezing')
        self.ws.finish_close('first', 'session1', self.checkpoint())
        with self.assertRaises(project.Invalid):
            store.init('other', 'contract', 'hypothesis', 'all')

    def test_paused_run_can_freeze_without_forging_terminal_result(self):
        store = search.Store(self.root, 'first', 'session1')
        store.init('group', 'contract', 'hypothesis', 'all')
        root = self.ws.project('first')
        task = root / 'TASK.md'
        task.write_text('- project_id: first\n- run_id: r1\n- node: analysis\n- search_node_id: N0\n')
        store.apply(dict(event_id='r1', op='run', legacy=True, id='r1', search_node_id='N0', stage='analysis',
                         candidate_id=None, task=str(task.relative_to(self.root))))
        self.ws.begin_close('first', 'session1', 'frozen')
        with self.assertRaises(project.Invalid):
            self.ws.finish_close('first', 'session1', self.checkpoint())
        (root / 'paused.md').write_text('- run_id: r1\n- status: interrupted\n')
        self.ws.finish_close('first', 'session1', self.checkpoint(paused_runs={'r1': 'paused.md'}))
        self.assertIsNone(store.load()['runs']['r1']['result'])
        self.ws.restore('first', 'session1', self.checkpoint(filename='lifecycle/restore.json'))
        self.assertIsNone(store.load()['runs']['r1']['result'])

    def test_missing_repo_or_changed_checkpoint_cannot_restore(self):
        source = self.repo()
        self.ws.add_repo('first', 'session1', 'operator', source, 'HEAD')
        evidence = self.checkpoint()
        self.ws.begin_close('first', 'session1', 'frozen')
        self.ws.finish_close('first', 'session1', evidence)
        repo = self.ws.project('first') / 'repos/operator'
        repo.rename(repo.with_name('moved'))
        with self.assertRaises(project.Invalid):
            self.ws.restore('first', 'session1', self.checkpoint(filename='lifecycle/restore.json'))
        repo.with_name('moved').rename(repo)
        (self.ws.project('first') / evidence).write_text('{}')
        with self.assertRaises(project.Invalid):
            self.ws.restore('first', 'session1', 'lifecycle/restore.json')
        self.assertEqual(self.ws.load('first')['status'], 'frozen')

    def test_search_projects_cannot_read_each_others_private_records(self):
        self.ws.create('second', 'Second', 'session2')
        a = search.Store(self.root, 'first', 'session1')
        b = search.Store(self.root, 'second', 'session2')
        a.init('same-name', 'contract', 'first hypothesis', 'all')
        b.init('same-name', 'contract', 'second hypothesis', 'all')
        self.assertNotEqual(a.path, b.path)
        with self.assertRaises(search.Invalid):
            a.local('.workspace/second/search/tree.json')
        with self.assertRaises(project.Invalid):
            search.Store(self.root, 'first', 'session2').apply(dict(event_id='e', op='select', id='N0', reason='test'))
        self.assertEqual(b.load()['nodes']['N0']['hypothesis'], 'second hypothesis')
        with self.assertRaises(project.Invalid):
            search.Store(self.root)

    def test_legacy_is_explicit_and_read_only(self):
        managed = search.Store(self.root, 'first', 'session1')
        tree = managed.init('example', 'contract', 'hypothesis', 'all')
        tree.pop('project_id')
        old = self.ws.directory / 'search/tree.json'
        old.parent.mkdir()
        old.write_text(json.dumps(tree))
        store = search.Store(self.root, legacy=True)
        self.assertEqual(store.load()['operator'], 'example')
        with self.assertRaisesRegex(search.Invalid, 'read-only'):
            store.init('example', 'contract', 'hypothesis', 'all')
        self.assertEqual(managed.load()['project_id'], 'first')

    def test_installer_keeps_explicit_nested_workspace_and_all_skills(self):
        source = self.repo()
        nested = source / 'workspace'
        nested.mkdir()
        command = [sys.executable, str(ROOT / 'scripts/install_codex_skills.py'), '--workspace', str(nested)]
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((source / '.agents').exists())
        installed = nested / '.agents/skills'
        self.assertEqual(len(list(installed.iterdir())), 13)
        self.assertTrue((installed / 'harness-port-baseline/assets/templates/porting.md').is_file())
        self.assertFalse((installed / 'harness-device-test').exists())
        measure = installed / 'harness-develop/scripts'
        smoke = subprocess.run([sys.executable, '-c',
                                'import sys; sys.path.insert(0, sys.argv[1]); '
                                'import torch_npu_measure; import profile_parser; '
                                'import torch_npu_diagnostics; '
                                'assert callable(torch_npu_measure.measure); '
                                'assert callable(torch_npu_diagnostics.capture_diagnostics)', str(measure)],
                               capture_output=True, text=True)
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        tool = installed / 'harness-new-project/scripts/projectctl.py'
        result = subprocess.run([sys.executable, str(tool), '--workspace', str(nested),
                                 '--project', 'example', '--owner', 'session', 'create', '--title', 'Example'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        search_tool = installed / 'harness-resume/scripts/searchctl.py'
        result = subprocess.run([sys.executable, str(search_tool), '--workspace', str(nested),
                                 '--project', 'example', '--owner', 'session', 'init', '--operator', 'group',
                                 '--task-contract', 'contract', '--hypothesis', 'test', '--scope', 'all'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('project_id', result.stdout)
        self.assertEqual((nested / 'AGENTS.md').read_text().count('<!-- operator-harness -->'), 1)

    def test_installer_rejects_stale_entry_before_copying(self):
        entry = self.root / 'AGENTS.md'
        old = '# Keep user notes\n<!-- operator-harness -->\nRead .workspace/STATE.md\n<!-- /operator-harness -->\n'
        entry.write_text(old)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/install_codex_skills.py'),
                                 '--workspace', str(self.root)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('AGENTS block differs', result.stderr)
        self.assertEqual(entry.read_text(), old)
        self.assertFalse((self.root / '.agents').exists())
        self.assertFalse((self.root / '.gitignore').exists())


if __name__ == '__main__':
    unittest.main()
