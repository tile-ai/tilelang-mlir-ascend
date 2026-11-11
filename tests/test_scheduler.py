"""Queue transitions with real locks/worktrees and the CPU evaluation pipeline."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_search_tools as fixtures
import test_execution_contract as execution
from records import render
search = fixtures.search
scheduler = search.scheduler


def concurrent_admit(root, operation):
    try:
        search.Store(root, 'example', 'session').apply(operation)
        return 'ok'
    except ValueError as exc:
        return str(exc)


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SearchToolsTest()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.store = self.f.store
        self.apply = self.f.apply
        self.source = self.f.root / 'origin'
        self.source.mkdir()
        for args in (('init',), ('config', 'user.name', 'Fixture'),
                     ('config', 'user.email', 'fixture@example.invalid'),
                     ('commit', '--allow-empty', '-m', 'fixture')):
            subprocess.run(['git', '-C', str(self.source), *args], check=True, capture_output=True)
        self.authority = self.file('authority.md', 'Fixture authorization')
        artifacts = {key: self.freeze(self.file(key + '.txt', key)) for key in ('source', 'build', 'launch')}
        self.manifest = self.file('C0.md', render('Candidate', {
            'contract': 'operator-harness/candidate@2', 'project_id': 'example',
            'candidate_id': 'C0', 'search_node_id': 'N0', 'parent_candidate': 'not_applicable',
            **{key: ref['path'] for key, ref in artifacts.items()},
            **{key + '_sha256': ref['sha256'] for key, ref in artifacts.items()}}))
        self.apply('candidate', id='C0', search_node_id='N0', parent_candidate=None, manifest=self.manifest)

    def file(self, name, text):
        return self.f.write('.workspace/example/' + name, text)

    def freeze(self, path):
        with self.store.write():
            return scheduler.frozen_blob(self.store, self.store.reference(path))

    def enable(self, slots=4, steps=12, budgets=None):
        self.apply('control-init', authority=self.authority, step_limit=steps, budgets=budgets or {})
        self.apply('queue-enable', max_parallel=slots, authority=self.authority)

    def repo(self, key):
        self.store.workspace.add_repo('example', 'session', key, self.source, 'HEAD')
        return key

    def enqueue(self, key, stage='analysis', **changes):
        spec = dict(stage=stage, operation='analyze' if stage == 'analysis' else 'code-only',
                    search_node_id='N0', candidate_id='C0', reason=key, inputs={}, requires=[])
        if stage == 'develop':
            spec['repo_ids'] = [self.repo(key)]
        spec.update(changes)
        self.apply('enqueue', id=key, spec=spec)
        return spec

    def admission(self, key, run=None, **fields):
        run = run or key
        item = self.store.load()['scheduling']['items'][key]
        values = dict(contract='operator-harness/node-task@4', project_id='example', run_id=run,
                      node=item['stage'], search_node_id=item['search_node_id'], queue_item_id=key,
                      operation=item['operation'], performance_required='false', diagnostics_required='false')
        values.update(fields)
        task = self.file('runs/' + run + '/TASK.md', render('Task', values))
        return dict(event_id='admit-' + key, op='queue-admit', id=key, revision=item['revision'],
                    run_id=run, request_id='dispatch-' + run, task=task, reason='chosen fixture')

    def admit(self, key, **fields):
        return self.store.apply(self.admission(key, **fields))

    def close(self, run_id):
        run = self.store.load()['runs'][run_id]
        checkpoint = self.file('runs/' + run_id + '/checkpoint.md', render('Checkpoint', {
            'contract': 'operator-harness/node-result@4', 'project_id': 'example',
            'run_id': run_id, 'node': run['stage'], 'search_node_id': run['search_node_id'], 'status': 'blocked'}))
        record = dict(schema='operator-harness/run-closure@1', project_id='example', run_id=run_id,
                      summary='fixture stopped', writers_stopped=True, jobs=[], unsettled_resources=[],
                      evidence=[self.store.reference(self.authority)])
        evidence = self.file('runs/' + run_id + '/close.json', json.dumps(record))
        return self.apply('close-run', id=run_id, checkpoint=checkpoint, evidence=evidence,
                          disposition='abandoned', reason='fixture stopped')

    def release(self, run_id, generation=None):
        lease = self.store.load()['scheduling']['leases'][run_id]
        generation = generation or lease['generation']
        record = dict(schema='operator-harness/workspace-release@1', project_id='example', run_id=run_id,
                      generation=generation, writers_stopped=True, jobs=[], unsettled_resources=[],
                      evidence=[self.store.reference(self.authority)])
        path = self.file('runs/' + run_id + '/release.json', json.dumps(record))
        return self.apply('release-workspace', id=run_id, generation=generation, evidence=path)

    def accept_analysis(self, run_id):
        run = self.store.load()['runs'][run_id]
        path = self.file('runs/' + run_id + '/RESULT.md', render('Result', {
            'contract': 'operator-harness/node-result@4', 'project_id': 'example', 'run_id': run_id,
            'node': 'analysis', 'search_node_id': run['search_node_id'],
            'queue_item_id': run['queue_item_id'], 'status': 'completed'}))
        return self.apply('accept', id=run_id, result=path)

    def states(self):
        return {row['id']: row for row in scheduler.view(self.store, self.store.load())['items']}

    def test_cpu_candidate_can_be_analyzed_while_sibling_develop_runs(self):
        self.enable(slots=3, budgets={'cpu': {'unit': 'seconds', 'limit': 10}})
        self.enqueue('sibling', 'develop')
        self.admit('sibling')
        self.repo('d')
        e = execution.ExecutionContractTest()
        e.f, e.store, e.prefix = self.f, self.store, '.workspace/example/runs/d/'
        original = self.f.apply
        def admit_pipeline(action, **values):
            if action != 'run':
                return original(action, **values)
            path = self.store.local(values['task'])
            from records import metadata
            fields = metadata(path.read_text())
            fields['queue_item_id'] = 'pipeline'
            path.write_text(render('Task', fields))
            original('enqueue', id='pipeline', spec=dict(stage='develop', operation='initial-and-validate',
                search_node_id='N0', candidate_id=None, repo_ids=['d'], reason='CPU integration'))
            return original('queue-admit', id='pipeline', revision=1, run_id='d', request_id='dispatch-d',
                            task=values['task'], reason='CPU integration')
        with patch.object(self.f, 'apply', side_effect=admit_pipeline):
            e.setup_pipeline()
        e.reservation = {'pool': 'cpu', 'amount': 5}
        produced = e.execute_pipeline()
        self.assertIn('result', produced, produced)
        self.apply('accept', id='d', result=produced['result'])
        self.release('d')
        self.enqueue('next', candidate_id='C1', requires=[{'run_id': 'd', 'condition': 'accepted'}])
        self.admit('next')
        self.store.local('.workspace/example/runs/d/candidate.py').write_text('changed original worktree')
        self.store.verify_run(self.store.load(), 'next')
        tree = self.store.load()
        self.assertIsNone(tree['runs']['sibling']['result'])
        self.assertEqual(tree['control']['steps'], ['sibling', 'd', 'next'])
        self.assertEqual(self.states()['next']['state'], 'in_flight')

    def test_dependencies_cycle_revision_and_cancelled_predecessor(self):
        self.enable()
        first = self.enqueue('a')
        self.enqueue('b', requires=[{'queue_item_id': 'a', 'condition': 'settled'}])
        self.enqueue('c', requires=[{'queue_item_id': 'a', 'condition': 'accepted'}])
        with self.assertRaisesRegex(search.Invalid, 'cycle'):
            self.apply('queue-revise', id='a', revision=1,
                       spec=dict(first, requires=[{'queue_item_id': 'b', 'condition': 'settled'}]))
        self.apply('queue-defer', id='a', revision=1, reason='prefer another direction')
        with self.assertRaisesRegex(search.Invalid, 'stale'):
            self.apply('queue-cancel', id='a', revision=1, reason='obsolete')
        self.apply('queue-cancel', id='a', revision=2, reason='obsolete')
        self.assertEqual(self.states()['b']['state'], 'ready')
        self.assertEqual(self.states()['c']['state'], 'blocked')
        self.assertEqual(self.store.load()['control']['steps'], [])

    def test_wait_all_requires_explicit_dependencies(self):
        self.enable()
        self.enqueue('a')
        self.enqueue('b', inputs={'question': self.freeze(self.file('question.txt', 'different question'))})
        self.admit('a')
        self.admit('b')
        self.enqueue('join', requires=[{'run_id': name, 'condition': 'settled'} for name in ('a', 'b')])
        self.accept_analysis('a')
        self.assertEqual(self.states()['join']['state'], 'blocked')
        self.close('b')
        self.admit('join')
        self.assertEqual(self.states()['join']['state'], 'in_flight')

    def test_worktree_reuse_requires_writer_release_and_generation(self):
        self.enable()
        spec = self.enqueue('d1', 'develop')
        self.admit('d1')
        self.apply('enqueue', id='d2', spec=dict(spec, reason='reuse'))
        self.assertIn('held', self.states()['d2']['reasons'][0])
        with self.assertRaisesRegex(search.Invalid, 'settled'):
            self.release('d1')
        self.close('d1')
        with self.assertRaisesRegex(search.Invalid, 'stale'):
            self.release('d1', 999)
        self.release('d1')
        self.admit('d2')
        leases = self.store.load()['scheduling']['leases']
        self.assertGreater(leases['d2']['generation'], leases['d1']['generation'])

    def test_atomic_last_slot_and_idempotent_retry(self):
        self.enable(slots=1)
        self.enqueue('a')
        self.enqueue('b', inputs={'q': self.freeze(self.file('q.txt', 'b'))})
        ops = [self.admission(key) for key in ('a', 'b')]
        with ProcessPoolExecutor(2) as pool:
            outcomes = list(pool.map(concurrent_admit, [str(self.f.root)] * 2, ops))
        self.assertEqual(outcomes.count('ok'), 1, outcomes)
        winner = ops[outcomes.index('ok')]
        before = self.store.path.read_bytes()
        self.store.apply(winner)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(len(self.store.load()['control']['steps']), 1)

    def test_snapshots_indices_and_drift(self):
        raw = self.file('mutable/source.py', 'old source')
        untracked = self.file('mutable/untracked.py', 'untracked dependency')
        index = self.file('index.json', json.dumps({'schema': 'operator-harness/artifact-index@1',
            'files': [self.store.reference(raw), self.store.reference(untracked)]}))
        frozen = self.freeze(index)
        mapping = json.loads(self.store.local(frozen['path']).read_text())['restore_paths']
        self.assertEqual(set(mapping), {raw, untracked})
        self.store.local(raw).write_text('new source')
        refs = scheduler.immutable_refs(self.store, [frozen])
        self.assertEqual(len(refs), 3)
        self.enable()
        with self.assertRaisesRegex(search.Invalid, 'snapshots'):
            self.enqueue('bad', inputs={'source': self.store.reference(raw)})
        self.enqueue('a', inputs={'source': frozen})
        child = next(ref for ref in refs.values() if ref['path'].endswith('/source.py'))
        self.store.local(child['path']).write_text('tampered snapshot')
        self.assertEqual(self.states()['a']['state'], 'blocked')

    def test_pause_steps_budget_and_physical_resource_exclusion(self):
        self.enable(steps=2, budgets={key: dict(unit='seconds', limit=10, exclusive_resource='device:test')
                                    for key in ('timing', 'profiling')})
        self.enqueue('a')
        self.enqueue('b', inputs={'q': self.freeze(self.file('q.txt', 'b'))})
        self.apply('control-pause', authority=self.authority)
        self.assertIn('paused', self.states()['a']['reasons'][0])
        self.apply('control-resume', authority=self.authority)
        self.admit('a')
        self.admit('b')
        self.apply('control-reserve', id='r1', run_id='a', pool='timing', amount=1)
        with self.assertRaisesRegex(search.Invalid, 'physical resource'):
            self.apply('control-reserve', id='r2', run_id='b', pool='profiling', amount=1)
        self.apply('control-settle', id='r1', used=0, evidence=self.authority)
        self.apply('control-reserve', id='r2', run_id='b', pool='profiling', amount=1)
        self.enqueue('c', inputs={'q': self.freeze(self.file('q2.txt', 'c'))})
        self.assertIn('step budget', self.states()['c']['reasons'][0])

    def test_legacy_safe_boundary_and_maintenance(self):
        self.f.record_run('old', stage='develop', candidate='C0', finish=False)
        with self.assertRaisesRegex(search.Invalid, 'safe boundary'):
            self.apply('queue-enable', max_parallel=2, authority=self.authority)
        artifacts = {key: self.freeze(self.file('legacy-' + key + '.txt', key)) for key in ('source', 'build', 'launch')}
        manifest = self.file('C_legacy.md', render('Candidate', {
            'contract': 'operator-harness/candidate@2', 'project_id': 'example',
            'candidate_id': 'C_legacy', 'search_node_id': 'N0', 'parent_candidate': 'C0',
            **{key: ref['path'] for key, ref in artifacts.items()},
            **{key + '_sha256': ref['sha256'] for key, ref in artifacts.items()}}))
        self.apply('candidate', id='C_legacy', search_node_id='N0', parent_candidate='C0',
                   producer_run='old', manifest=manifest)
        result = self.f.write('.workspace/example/runs/old/RESULT.md',
                              '- project_id: example\n- run_id: old\n- node: develop\n- search_node_id: N0\n'
                              '- operation: implement-and-validate\n- performance_required: true\n'
                              '- status: completed\n- produced_candidate: .workspace/example/C_legacy.md\n'
                              + self.f.acceptance_fields('old'))
        self.apply('accept', id='old', result=result)
        self.enable()
        self.enqueue('a', candidate_id='C_legacy')
        self.admit('a')
        self.f.add_node()
        self.apply('select', id='N1', reason='navigation only')
        with self.assertRaisesRegex(search.Invalid, 'queue-admit'):
            self.f.record_run('b', candidate='C0', finish=False)
        with self.assertRaisesRegex(search.Invalid, 'writers'):
            self.apply('run', id='maintenance', stage='prepare-environment')
        self.accept_analysis('a')
        with self.assertRaisesRegex(search.Invalid, 'writers'):
            self.apply('node-status', id='N0', status='closed', reason='finished')
        self.release('a')
        self.apply('node-status', id='N0', status='closed', reason='finished')

    def test_remote_alias_overlap_uses_confirmed_domain(self):
        self.enable()
        def remote(path):
            ref = self.store.reference(self.file('remote' + path.replace('/', '_') + '.json', json.dumps({
                'schema': 'operator-harness/workspace-identity@1', 'domain': 'remote:physical-host:mount',
                'path': path, 'canonical': True, 'source': 'fixture realpath query'})))
            return dict(domain='remote:physical-host:mount', path=path, identity_evidence=ref)
        self.enqueue('a', workspaces=[remote('/build/task')])
        self.admit('a')
        self.enqueue('b', workspaces=[remote('/build/task/cache')])
        self.assertIn('held', self.states()['b']['reasons'][0])
        self.enqueue('c', workspaces=[remote('/build/other')])
        self.admit('c')

    def test_comparison_group_binds_contract_without_wait_barrier(self):
        self.enable()
        inputs = {key: self.freeze(self.file(key + '.txt', key)) for key in
                  ('requirements', 'reference', 'comparator', 'cases', 'protocol', 'environment')}
        group = {key: inputs[key] for key in ('requirements', 'cases', 'protocol', 'environment')}
        group['baseline'] = inputs['protocol']  # explicit no-performance scope
        specs = []
        for key in ('v1', 'v2'):
            specs.append(self.enqueue(key, stage='develop', operation='verify-only'))
        self.apply('comparison-group', id='G1', members=['v1', 'v2'], inputs=group)
        adapter = self.freeze(self.file('adapter.py', '# Frozen queue-only adapter fixture\n'))
        for key in ('v1', 'v2'):
            plan = dict(schema='operator-harness/evaluation-plan@1', project_id='example', run_id=key,
                        inputs=inputs, performance_kind='none', statistic='mean',
                        execution={'backend': 'cpu', 'adapter': adapter, 'timeout_seconds': 1},
                        cases=[{'id': 'one', 'outputs': ['out']}])
            if key == 'v2':
                plan['cases'][0]['outputs'] = ['out', 'count']
            frozen = self.freeze(self.file(key + '-plan.json', json.dumps(plan)))
            if key == 'v1':
                self.admit(key, evaluation_plan=frozen['path'])
            else:
                with self.assertRaisesRegex(search.Invalid, 'comparison plans differ'):
                    self.admit(key, evaluation_plan=frozen['path'])
        self.assertIsNone(self.store.load()['runs']['v1']['result'])
        self.assertNotIn('v2', self.store.load()['runs'])
        with self.assertRaisesRegex(search.Invalid, 'parent/stage'):
            self.apply('queue-revise', id='v2', revision=1,
                       spec=dict(specs[1], candidate_id=None, operation='initial-and-validate'))

    def test_active_and_closed_producer_requires_failure_diagnosis(self):
        self.enable()
        self.enqueue('writer', 'develop')
        self.admit('writer')
        artifacts = [self.file('runs/writer/' + kind, kind) for kind in ('source.py', 'build.json', 'launch.json')]
        execution.evaluate.candidate(self.store, 'writer', 'C1', *artifacts, 'C0')
        spec = self.enqueue('reader', candidate_id='C1')
        self.assertIn('not settled', self.states()['reader']['reasons'][0])
        self.close('writer')
        self.assertIn('failure diagnosis', self.states()['reader']['reasons'][0])
        self.apply('queue-revise', id='reader', revision=1, spec=dict(spec, operation='diagnose-failure'))
        self.admit('reader')
        self.release('writer')
        with self.assertRaisesRegex(search.Invalid, 'settled or released'):
            execution.evaluate.candidate(self.store, 'writer', 'C2', *artifacts, 'C0')

    def test_freeze_project_preserves_confirmed_workspace_generation(self):
        self.enable()
        self.enqueue('a')
        self.admit('a')
        self.close('a')
        workspace = self.store.workspace
        workspace.begin_close('example', 'session', 'frozen')
        checkpoint = dict(schema='operator-harness/project-checkpoint@1', project_id='example',
                          summary='fixture stopped writers', writers_stopped=True, jobs=[],
                          unsettled_resources=[], open_runs=[])
        path = self.file('lifecycle/freeze.json', json.dumps(checkpoint))
        with self.assertRaisesRegex(search.projectctl.Invalid, 'workspace generation'):
            workspace.finish_close('example', 'session', 'lifecycle/freeze.json')
        checkpoint['stopped_workspace_leases'] = {'a': 1}
        self.store.local(path).write_text(json.dumps(checkpoint))
        result = workspace.finish_close('example', 'session', 'lifecycle/freeze.json')
        self.assertEqual(result['status'], 'frozen')
        self.assertFalse(self.store.load()['scheduling']['leases']['a']['released'])

    def test_invalid_plan_admission_has_no_step_or_lease_side_effects(self):
        self.enable()
        self.enqueue('v', stage='develop', operation='initial-and-validate', candidate_id=None)
        bad = self.file('plan.json', '{}')
        before = self.store.path.read_bytes()
        with self.assertRaisesRegex(search.Invalid, 'evaluation plan'):
            self.admit('v', evaluation_plan=bad)
        valid_plan = self.file('runs/v/plan.json', json.dumps(dict(
            schema='operator-harness/evaluation-plan@1', project_id='example', run_id='v',
            inputs={key: self.freeze(self.file('in-' + key + '.txt', key)) for key in
                    ('requirements', 'reference', 'comparator', 'cases', 'protocol', 'environment')},
            execution={'backend': 'cpu', 'timeout_seconds': 1,
                       'adapter': self.freeze(self.file('adapter.py', '# Frozen queue-only adapter fixture\n'))},
            performance_kind='none', statistic='mean', cases=[{'id': 'one', 'outputs': ['out']}])))
        mutable_design = self.file('repos/v/design.md', 'mutable design')
        with self.assertRaisesRegex(search.Invalid, 'mutable worktree/build'):
            self.admit('v', evaluation_plan=valid_plan, initial_design=mutable_design)
        self.assertEqual(before, self.store.path.read_bytes())
        self.assertEqual(self.store.load()['control']['steps'], [])


if __name__ == '__main__':
    unittest.main()
