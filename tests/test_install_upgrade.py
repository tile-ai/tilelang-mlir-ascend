"""Managed upgrade/rollback and interruption checks in isolated installations."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import skill_bundle as bundle


class InstallUpgradeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source'
        (self.source / 'harness-example').mkdir(parents=True)
        (self.source / 'harness-example/SKILL.md').write_text('old')
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.name = '.agents/skills/harness-example/SKILL.md'
        target = self.workspace / self.name
        target.parent.mkdir(parents=True)
        target.write_text('old')
        self.entry = (bundle.START + '\nold\n' + bundle.END).encode()
        (self.workspace / 'AGENTS.md').write_bytes(b'User instructions\n' + self.entry + b'\nTail\n')
        bundle.record_install(self.workspace, self.source, ['harness-example'], self.entry)
        (target.parent / 'personal.md').write_text('keep me')
        (self.source / 'harness-example/SKILL.md').write_text('new')
        self.new_entry = self.entry.replace(b'old', b'new')

    def prepare(self):
        return bundle.prepare(self.workspace, self.source, ['harness-example'], self.new_entry)[0]

    def apply(self, plan):
        return bundle.upgrade(self.workspace, self.source, ['harness-example'], self.new_entry, plan['id'])

    def test_upgrade_and_rollback_preserve_unmanaged_content(self):
        plan = self.prepare()
        self.apply(plan)
        self.assertEqual((self.workspace / self.name).read_text(), 'new')
        self.assertIn('User instructions', (self.workspace / 'AGENTS.md').read_text())
        self.assertEqual((self.workspace / self.name).with_name('personal.md').read_text(), 'keep me')
        bundle.transaction(self.workspace, plan['id'], rollback=True)
        self.assertEqual((self.workspace / self.name).read_text(), 'old')
        self.assertNotEqual(self.prepare()['id'], plan['id'])

    def test_user_modified_file_rejects_plan_without_writes(self):
        (self.workspace / self.name).write_text('user edit')
        before = (self.workspace / 'AGENTS.md').read_bytes()
        with self.assertRaisesRegex(ValueError, 'user-modified'):
            self.prepare()
        self.assertEqual(before, (self.workspace / 'AGENTS.md').read_bytes())

    def test_interrupted_upgrade_resumes_without_source_bundle(self):
        plan = self.prepare()
        original = bundle.atomic
        def fail(path, data):
            if path == self.workspace / 'AGENTS.md':
                raise OSError('synthetic interruption')
            return original(path, data)
        with patch.object(bundle, 'atomic', side_effect=fail), self.assertRaises(OSError):
            self.apply(plan)
        self.assertEqual((self.workspace / self.name).read_text(), 'new')
        bundle.transaction(self.workspace, plan['id'])
        self.assertIn('\nnew\n', (self.workspace / 'AGENTS.md').read_text())

    def test_rollback_refuses_to_overwrite_post_upgrade_edits(self):
        plan = self.prepare()
        self.apply(plan)
        (self.workspace / self.name).write_text('new user edit')
        with self.assertRaisesRegex(ValueError, 'changed after'):
            bundle.transaction(self.workspace, plan['id'], rollback=True)
        self.assertIn('\nnew\n', (self.workspace / 'AGENTS.md').read_text())

    def test_source_linked_workspace_rejects_before_claiming_managed_files(self):
        import install_codex_skills as installer
        linked = self.root / 'linked'
        target = linked / '.agents/skills'
        target.mkdir(parents=True)
        (target / installer.NAMES[0]).symlink_to(installer.SOURCE / installer.NAMES[0], target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'source-linked'):
            installer.install(linked)
        self.assertFalse((linked / '.agents/.harness/manifest.json').exists())

    def test_initial_install_rejects_symlinked_ancestors_before_any_writes(self):
        import install_codex_skills as installer
        for relative in ('.agents', '.agents/skills'):
            for location in ('outside', 'inside', 'dangling'):
                with self.subTest(relative=relative, location=location), tempfile.TemporaryDirectory() as temporary:
                    base = Path(temporary)
                    workspace = base / 'workspace'
                    workspace.mkdir()
                    shared = workspace / 'shared' if location == 'inside' else base / 'shared'
                    if location != 'dangling':
                        shared.mkdir()
                    link = workspace / relative
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(shared, target_is_directory=True)
                    before = {str(p.relative_to(base)) for p in base.rglob('*')}
                    result = subprocess.run([sys.executable, installer.__file__, '--workspace', str(workspace)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn('symlink', result.stderr)
                    self.assertEqual({str(p.relative_to(base)) for p in base.rglob('*')}, before)

    def test_initial_install_rejects_linked_output_files_before_copying(self):
        import install_codex_skills as installer
        for relative in ('AGENTS.md', '.gitignore', '.agents/.harness/manifest.json', '.agents/.harness/lock'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                workspace = base / 'workspace'
                workspace.mkdir()
                external = base / 'external'
                external.write_text('user content\n')
                link = workspace / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(external)
                before = {str(p.relative_to(base)) for p in base.rglob('*')}
                result = subprocess.run([sys.executable, installer.__file__, '--workspace', str(workspace)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('symlink', result.stderr)
                self.assertEqual(external.read_text(), 'user content\n')
                self.assertEqual({str(p.relative_to(base)) for p in base.rglob('*')}, before)

    def test_initial_install_checks_nested_managed_paths(self):
        import install_codex_skills as installer
        workspace = self.root / 'nested-links'
        skills = workspace / '.agents/skills'
        shutil.copytree(installer.SOURCE, skills, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        references = skills / 'harness-new-project/references'
        outside = self.root / 'shared-references'
        references.rename(outside)
        references.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            installer.install(workspace)
        self.assertFalse((workspace / 'AGENTS.md').exists())
        self.assertFalse((workspace / '.gitignore').exists())
        self.assertFalse((workspace / '.agents/.harness').exists())

    def test_upgrade_and_rollback_reject_aliased_transaction_history(self):
        plan = self.prepare()
        self.apply(plan)
        history = self.workspace / '.agents/.harness/transactions'
        outside = self.root / 'shared-transactions'
        history.rename(outside)
        history.symlink_to(outside, target_is_directory=True)
        contents = {p.name: p.read_bytes() for p in outside.iterdir()}
        installed = (self.workspace / self.name).read_bytes()
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'symlink'):
            bundle.transaction(self.workspace, plan['id'], rollback=True)
        self.assertEqual({p.name: p.read_bytes() for p in outside.iterdir()}, contents)
        self.assertEqual((self.workspace / self.name).read_bytes(), installed)

    def test_changed_bundle_invalidates_reviewed_plan(self):
        plan = self.prepare()
        (self.source / 'harness-example/SKILL.md').write_text('different')
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            self.apply(plan)
        self.assertEqual((self.workspace / self.name).read_text(), 'old')

    def test_removed_managed_file_is_deleted_but_extra_file_survives(self):
        (self.source / 'harness-example/SKILL.md').unlink()
        self.apply(self.prepare())
        self.assertFalse((self.workspace / self.name).exists())
        self.assertEqual((self.workspace / self.name).with_name('personal.md').read_text(), 'keep me')

    def test_mfu_to_roofline_rename_and_rollback_preserve_user_files(self):
        old = self.source / 'ascend-mfu/SKILL.md'
        old.parent.mkdir()
        old.write_text('old MFU')
        target = self.workspace / '.agents/skills/ascend-mfu/SKILL.md'
        target.parent.mkdir(parents=True)
        target.write_bytes(old.read_bytes())
        personal = target.with_name('personal.md')
        personal.write_text('user notes')
        (bundle.metadata(self.workspace) / 'manifest.json').unlink()
        bundle.record_install(self.workspace, self.source, ['ascend-mfu'], self.entry)
        new = self.source / 'ascend-roofline/SKILL.md'
        new.parent.mkdir()
        new.write_text('merged Roofline')
        names = ['ascend-roofline']
        target.write_text('user edit')
        with self.assertRaisesRegex(ValueError, 'user-modified'):
            bundle.prepare(self.workspace, self.source, names, self.new_entry)
        target.write_bytes(old.read_bytes())
        plan = bundle.prepare(self.workspace, self.source, names, self.new_entry)[0]
        bundle.upgrade(self.workspace, self.source, names, self.new_entry, plan['id'])
        installed = self.workspace / '.agents/skills/ascend-roofline/SKILL.md'
        self.assertFalse(target.exists())
        self.assertEqual(installed.read_text(), 'merged Roofline')
        self.assertEqual(personal.read_text(), 'user notes')
        bundle.transaction(self.workspace, plan['id'], rollback=True)
        self.assertEqual(target.read_text(), 'old MFU')
        self.assertFalse(installed.exists())
        self.assertEqual(personal.read_text(), 'user notes')

    def test_legacy_adoption_requires_exact_old_bundle(self):
        legacy = self.root / 'legacy'
        skill = legacy / 'assets/skills/harness-example/SKILL.md'
        skill.parent.mkdir(parents=True)
        skill.write_text('wrong')
        entry = legacy / 'assets/codex/AGENTS.md'
        entry.parent.mkdir(parents=True)
        entry.write_bytes(self.entry)
        manifest = bundle.metadata(self.workspace) / 'manifest.json'
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, 'does not match'):
            bundle.adopt(self.workspace, legacy)
        self.assertFalse(manifest.exists())
        skill.write_text('old')
        self.assertEqual(bundle.adopt(self.workspace, legacy)['status'], 'adopted')
        self.apply(self.prepare())
        self.assertEqual((self.workspace / self.name).read_text(), 'new')

    def test_interrupted_rollback_resumes_in_rollback_direction(self):
        plan = self.prepare()
        self.apply(plan)
        original = bundle.atomic
        def fail(path, data):
            if path == self.workspace / 'AGENTS.md':
                raise OSError('synthetic rollback interruption')
            return original(path, data)
        with patch.object(bundle, 'atomic', side_effect=fail), self.assertRaises(OSError):
            bundle.transaction(self.workspace, plan['id'], rollback=True)
        self.assertEqual((self.workspace / self.name).read_text(), 'old')
        result = bundle.transaction(self.workspace, plan['id'])
        self.assertEqual(result['status'], 'rolled_back')
        self.assertIn('\nold\n', (self.workspace / 'AGENTS.md').read_text())


if __name__ == '__main__':
    unittest.main()
