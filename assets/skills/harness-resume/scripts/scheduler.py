"""Dependency admission and cooperative workspace leases; never dispatches agents.

All mutations run inside Store.apply's project/search transaction. Queue readiness
is a projection, not another execution state machine. Frozen blobs may precede
their ledger reference, so interruptions leave reusable, content-addressed files.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re

import control
import evidence
from evidence import Invalid, require

SCHEMA = 'operator-harness/search@2'
OPERATIONS = {'queue-enable', 'enqueue', 'queue-revise', 'queue-cancel', 'queue-defer',
              'queue-admit', 'release-workspace', 'comparison-group'}


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value),
            'invalid queue identifier')
    return value


def text(value, label):
    require(isinstance(value, str) and value.strip(), 'missing ' + label)
    return value


def settled(run):
    return bool(run.get('result') or run.get('closure'))


def overlap(a, b):
    return a == b or a.startswith(b.rstrip('/') + '/') or b.startswith(a.rstrip('/') + '/')


def frozen_blob(store, ref):
    """Copy an explicit file/index before constructing a candidate or TASK."""
    dependencies = {}
    evidence.freeze(store, ref, dependencies)  # validate the complete index first
    memo, visiting = {}, set()
    def copy(item):
        if item['path'] in memo:
            return memo[item['path']]
        require(item['path'] not in visiting, 'snapshot index cycle')
        visiting.add(item['path'])
        store.verify_ref(item)
        source = store.local(item['path'])
        payload = source.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'source changed while freezing')
        try:
            data = json.loads(payload) if source.suffix == '.json' else None
        except (ValueError, UnicodeError):
            data = None
        if isinstance(data, dict) and data.get('schema') in (
                'operator-harness/artifact-index@1', 'operator-harness/evaluation-plan@1',
                'operator-harness/diagnostic-evidence@1'):
            if data['schema'] == 'operator-harness/artifact-index@1' and 'restore_paths' not in data:
                data['restore_paths'] = {child['path']: copy(child)['path'] for child in data['files']}
            def rewrite(value):
                if isinstance(value, dict):
                    if set(value) == {'path', 'sha256'}:
                        return copy(value)
                    return {key: rewrite(child) for key, child in value.items()}
                if isinstance(value, list):
                    return [rewrite(child) for child in value]
                return value
            data = rewrite(data)
            payload = (json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
        sha = hashlib.sha256(payload).hexdigest()
        destination = store.records / 'snapshots' / sha / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(destination.read_bytes() == payload, 'frozen blob changed')
        else:
            # Atomic link publishes a complete file without replacing an existing blob.
            import os
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=destination.parent, prefix='.pending-')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(tmp, destination)
            finally:
                os.unlink(tmp)
        memo[item['path']] = store.reference(str(destination.relative_to(store.root)))
        visiting.remove(item['path'])
        return memo[item['path']]
    return copy(ref)


def immutable_refs(store, refs):
    """Require recursive source/adapter indices to live outside writable worktrees."""
    dependencies = {}
    for ref in refs:
        evidence.freeze(store, ref, dependencies)
    root = (store.records / 'snapshots').resolve()
    for ref in dependencies.values():
        path = store.local(ref['path'])
        require(path.is_relative_to(root), 'queue input must be frozen under project snapshots: ' + ref['path'])
        require(path.parent.name == ref['sha256'], 'snapshot path must be content addressed: ' + ref['path'])
    return dependencies


def stable_refs(store, refs):
    """Allow snapshots or run/environment records; reject mutable worktrees/builds."""
    dependencies = {}
    for ref in refs:
        evidence.freeze(store, ref, dependencies)
    mutable = ((store.records / 'repos').resolve(), (store.records / 'builds').resolve())
    snapshots = (store.records / 'snapshots').resolve()
    for ref in dependencies.values():
        path = store.local(ref['path'])
        require(not any(path.is_relative_to(m) for m in mutable),
                'evaluation input must not live in a mutable worktree/build: ' + ref['path'])
        if path.is_relative_to(snapshots):
            require(path.parent.name == ref['sha256'], 'snapshot path must be content addressed: ' + ref['path'])
    return dependencies


def candidate_inputs(store, tree, candidate_id):
    candidate = tree['candidates'][candidate_id]['manifest']
    store.verify_ref(candidate)
    artifacts = {key: store.reference(store.value(candidate, key)) for key in ('source', 'build', 'launch')}
    for key, ref in artifacts.items():
        require(ref['sha256'] == store.value(candidate, key + '_sha256'), 'candidate ' + key + ' changed')
    refs = immutable_refs(store, artifacts.values())
    refs[candidate['path']] = candidate
    return refs


def active_leases(tree):
    return [lease for lease in tree.get('scheduling', {}).get('leases', {}).values() if not lease.get('released')]


def validate(tree):
    scheduling = tree.get('scheduling')
    require((tree['schema'] == SCHEMA) == (scheduling is not None), 'queue/schema mismatch')
    if scheduling is None:
        return
    require(type(scheduling['max_parallel']) is int and scheduling['max_parallel'] > 0, 'invalid parallel limit')
    items = scheduling['items']
    for key, item in items.items():
        identifier(key)
        require(item['search_node_id'] in tree['nodes'], 'unknown queue node')
        require(item['stage'] in ('develop', 'analysis'), 'queue supports analysis/develop')
        require(type(item['revision']) is int and item['revision'] > 0, 'invalid queue revision')
        require(item['state'] in ('queued', 'admitted', 'cancelled'), 'invalid queue state')
        if item['state'] == 'admitted':
            run = tree['runs'][item['run_id']]
            require(run.get('queue_item_id') == key and run['stage'] == item['stage'] and
                    run['search_node_id'] == item['search_node_id'], 'queue/run mismatch')
        for dep in item['requires']:
            require(set(dep) in ({'run_id', 'condition'}, {'queue_item_id', 'condition'}), 'invalid dependency')
            require(dep['condition'] in ('accepted', 'settled'), 'invalid dependency condition')
            if 'run_id' in dep:
                require(dep['run_id'] in tree['runs'], 'unknown dependency run')
            else:
                require(dep['queue_item_id'] in items, 'unknown dependency queue item')
    def visit(key, stack, done):
        require(key not in stack, 'queue dependency cycle')
        if key in done:
            return
        stack.add(key)
        for dep in items[key]['requires']:
            target = dep.get('queue_item_id') or tree['runs'].get(dep.get('run_id'), {}).get('queue_item_id')
            if target:
                visit(target, stack, done)
        stack.remove(key)
        done.add(key)
    done = set()
    for key in items:
        visit(key, set(), done)
    requests = []
    for key, run in tree['runs'].items():
        if run.get('queue_item_id'):
            item = items[run['queue_item_id']]
            require(item.get('run_id') == key and key in scheduling['leases'], 'unlisted queue run/lease')
            requests.append(run['dispatch_request_id'])
    require(len(requests) == len(set(requests)), 'duplicate dispatch request')
    for run_id, lease in scheduling['leases'].items():
        require(run_id in tree['runs'] and type(lease['generation']) is int and
                lease['generation'] > 0 and type(lease['released']) is bool, 'invalid workspace lease')
        if lease['released']:
            require(settled(tree['runs'][run_id]) and lease.get('evidence'), 'released unsettled lease')
    for group in scheduling['groups'].values():
        require(len(group['members']) >= 2 and len(group['members']) == len(set(group['members'])) and
                all(member in items for member in group['members']), 'invalid comparison group')
        require(all(items[member]['candidate_id'] == group['candidate_id'] and
                    items[member]['stage'] == 'develop' for member in group['members']),
                'comparison parent/stage is fixed')
    leases = active_leases(tree)
    require(len(leases) <= scheduling['max_parallel'], 'active leases exceed parallel limit')
    for index, lease in enumerate(leases):
        for other in leases[index + 1:]:
            require(not any(a['domain'] == b['domain'] and overlap(a['path'], b['path'])
                            for a in lease['resources'] for b in other['resources']), 'overlapping active workspace leases')


def _verify_run_cached(store, tree, run_id, cache):
    if run_id not in cache['runs']:
        store.verify_run(tree, run_id)
        cache['runs'].add(run_id)


def dependencies(store, tree, item, cache=None):
    cache = cache if cache is not None else {'runs': set(), 'repos': set()}
    refs = {}
    for dep in item['requires']:
        if 'queue_item_id' in dep:
            upstream = tree['scheduling']['items'][dep['queue_item_id']]
            if upstream['state'] == 'cancelled' and dep['condition'] == 'settled':
                continue
            require(upstream['state'] == 'admitted', 'dependency queue item not admitted: ' + dep['queue_item_id'])
            run_id = upstream['run_id']
        else:
            run_id = dep['run_id']
        run = tree['runs'][run_id]
        require(bool(run.get('result')) if dep['condition'] == 'accepted' else settled(run),
                'dependency not ' + dep['condition'] + ': ' + run_id)
        control.reconciled(tree, run_id)
        if run.get('result'):
            _verify_run_cached(store, tree, run_id, cache)
            # A failed analysis RESULT may be accepted for bookkeeping, but cannot
            # satisfy an accepted dependency for implementation.
            if dep['condition'] == 'accepted':
                store.markdown(run['result'], status='completed')
            refs[run['result']['path']] = run['result']
        else:
            store.validate_closure(run_id, run)
            refs.update({ref['path']: ref for ref in
                         (run['closure']['checkpoint'], run['closure']['evidence'])})
    return refs


def workspace_resources(store, tree, item, run_id, cache=None):
    cache = cache if cache is not None else {'runs': set(), 'repos': set()}
    project = store.workspace.load(store.project_id)
    resources = []
    repos = item['repo_ids']
    require(len(repos) == len(set(repos)), 'duplicate repo_id')
    for key in repos:
        require(key in project['repositories'], 'unregistered worktree')
        repo = project['repositories'][key]
        if key not in cache['repos']:
            store.workspace.inventory(store.project_id, {'repositories': {key: repo}})
            cache['repos'].add(key)
        path = store.workspace.local(store.project_id, repo['path']).resolve()
        resources.append({'domain': 'local', 'path': str(path.relative_to(store.root))})
    for resource in item['workspaces']:
        require(set(resource) in ({'domain', 'path'}, {'domain', 'path', 'identity_evidence'}), 'invalid workspace resource')
        domain, path = text(resource['domain'], 'workspace domain'), text(resource['path'], 'workspace path')
        if domain == 'local':
            path = store.local(path)
            allowed = store.records / 'builds'
            require(path.is_relative_to(allowed), 'extra local workspace must be under project builds')
            path = str(path.relative_to(store.root))
        else:
            require(domain.startswith('remote:'), 'remote domain must identify canonical machine/container/mount')
            require(PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts and path != '/',
                    'remote directory must be a bounded absolute path')
            require('identity_evidence' in resource, 'remote workspace needs identity evidence')
            store.verify_ref(resource['identity_evidence'])
            identity = json.loads(store.local(resource['identity_evidence']['path']).read_text())
            require(identity.get('schema') == 'operator-harness/workspace-identity@1' and
                    identity.get('domain') == domain and identity.get('path') == str(PurePosixPath(path)) and
                    identity.get('canonical') is True and identity.get('source'),
                    'remote workspace identity must confirm canonical domain/path')
            path = str(PurePosixPath(path))
        resources.append({**resource, 'domain': domain, 'path': path})
    if run_id:
        resources.append({'domain': 'local', 'path': str((store.records / 'runs' / run_id).relative_to(store.root))})
    for index, resource in enumerate(resources):
        require(not any(resource['domain'] == other['domain'] and overlap(resource['path'], other['path'])
                        for other in resources[:index]), 'workspaces overlap within task')
        for lease in active_leases(tree):
            require(not any(resource['domain'] == other['domain'] and overlap(resource['path'], other['path'])
                            for other in lease['resources']), 'workspace held by another run')
    return resources


def checks(store, tree, item, run_id=None, cache=None):
    cache = cache if cache is not None else {'runs': set(), 'repos': set()}
    require(item['state'] == 'queued', 'queue item is not queued')
    require(tree['nodes'][item['search_node_id']]['status'] == 'open', 'queue strategy is not open')
    require(not any(not settled(run) and not run.get('queue_item_id') for run in tree['runs'].values()),
            'maintenance run is outstanding')
    refs = dependencies(store, tree, item, cache)
    refs.update(immutable_refs(store, item['inputs'].values()))
    candidate_id = item['candidate_id']
    require(candidate_id is not None or item['stage'] == 'develop' and
            item['operation'] in ('initial-and-validate', 'code-only'), 'bind a candidate before admission')
    if candidate_id is not None:
        require(candidate_id in tree['candidates'], 'unknown queue candidate')
        refs.update(candidate_inputs(store, tree, candidate_id))
    analysis_id = item.get('analysis_run')
    if item['stage'] == 'develop' and item['operation'] == 'implement-and-validate':
        require(analysis_id in tree['runs'], 'implementation needs an accepted analysis')
        analysis = tree['runs'][analysis_id]
        require(analysis['stage'] == 'analysis' and analysis['candidate_id'] == candidate_id and
                analysis.get('result'), 'analysis candidate/status mismatch')
        _verify_run_cached(store, tree, analysis_id, cache)
        store.markdown(analysis['result'], status='completed')
        refs[analysis['result']['path']] = analysis['result']
    # Registered/frozen is not the same as completed. A candidate still produced
    # by an active writer cannot enter a subsequent normal analysis/develop.
    if candidate_id:
        manifest = tree['candidates'][candidate_id]['manifest']
        values = store.local(manifest['path']).read_text()
        from records import metadata
        producer = metadata(values).get('producer_run')
        if producer:
            require(producer in tree['runs'] and settled(tree['runs'][producer]), 'candidate producer has not settled')
            origin = tree['runs'][producer]
            if origin.get('closure'):
                require(item['stage'] == 'analysis' and item['operation'] == 'diagnose-failure',
                        'closed candidate requires explicit failure diagnosis')
                store.validate_closure(producer, origin)
                refs[origin['closure']['evidence']['path']] = origin['closure']['evidence']
            else:
                _verify_run_cached(store, tree, producer, cache)
                acceptance = origin.get('acceptance', {})
                output = acceptance.get('validation', {}).get('candidate_id')
                if output is None and acceptance.get('evidence', {}).get('candidate'):
                    output = store.value(acceptance['evidence']['candidate'], 'candidate_id')
                require(output == candidate_id or (output is None and origin.get('legacy', 'evaluator' not in origin)),
                        'candidate is not the accepted producer output')
                refs[origin['result']['path']] = origin['result']
    c = tree.get('control', {})
    require(not c.get('paused'), 'control is paused')
    require(not c or len(c['steps']) < c['step_limit'], 'control step budget exhausted')
    for pool, amount in item['budget_requirements'].items():
        require(pool in c.get('budgets', {}), 'unknown budget pool: ' + pool)
        require(evidence.number(amount, positive=True) and control.available(c, pool) >= amount,
                'insufficient requested budget: ' + pool)
    require(len(active_leases(tree)) < tree['scheduling']['max_parallel'], 'parallel slots exhausted (release settled writers)')
    resources = workspace_resources(store, tree, item, run_id, cache)
    return refs, resources


def view(store, tree):
    if tree.get('schema') != SCHEMA:
        return {'enabled': False}
    rows = []
    cache = {'runs': set(), 'repos': set()}
    for key, item in tree['scheduling']['items'].items():
        state, reasons = item['state'], []
        if state == 'queued':
            try:
                checks(store, tree, item, cache=cache)
                state = 'ready'
            except (ValueError, OSError, KeyError, TypeError) as exc:
                state, reasons = 'blocked', [str(exc)]
        elif state == 'admitted':
            run = tree['runs'][item['run_id']]
            state = 'accepted' if run.get('result') else 'closed' if run.get('closure') else 'in_flight'
        rows.append({'id': key, 'revision': item['revision'], 'state': state,
                     'priority': item['priority'], 'search_node_id': item['search_node_id'],
                     'run_id': item.get('run_id'), 'reasons': reasons,
                     'reason': item['reason'], 'deferrals': item.get('deferrals', [])})
    return {'enabled': True, 'max_parallel': tree['scheduling']['max_parallel'],
            'items': sorted(rows, key=lambda row: (-row['priority'], row['id'])),
            'held_workspaces': {key: value for key, value in tree['scheduling']['leases'].items() if not value['released']}}


def proposal(store, tree, spec):
    allowed = {'stage', 'operation', 'search_node_id', 'candidate_id', 'analysis_run', 'plan_id',
               'mechanism_id', 'requires', 'inputs', 'repo_ids', 'workspaces', 'priority',
               'reason', 'budget_requirements', 'supersedes'}
    require(isinstance(spec, dict) and set(spec) <= allowed, 'unknown queue proposal fields')
    item = deepcopy(spec)
    require(item['stage'] in ('analysis', 'develop'), 'queue supports analysis/develop')
    require(item['search_node_id'] in tree['nodes'], 'unknown queue strategy')
    modes = ('analyze', 'diagnose-failure') if item['stage'] == 'analysis' else (
        'implement-and-validate', 'initial-and-validate', 'verify-only', 'collect-only', 'code-only')
    require(item['operation'] in modes, 'invalid queue operation')
    text(item.get('reason'), 'queue reason')
    item.setdefault('candidate_id', None)
    require(item['candidate_id'] is None or item['candidate_id'] in tree['candidates'], 'unknown candidate')
    for key, default in (('requires', []), ('inputs', {}), ('repo_ids', []), ('workspaces', []),
                         ('priority', 0), ('budget_requirements', {})):
        item.setdefault(key, default)
        require(type(item[key]) is type(default), 'invalid queue ' + key)
    if item['stage'] == 'develop' and item['operation'] in ('implement-and-validate', 'initial-and-validate', 'code-only'):
        require(bool(item['repo_ids']), 'writable develop needs an independent registered worktree')
    for repo in item['repo_ids']:
        identifier(repo)
    immutable_refs(store, item['inputs'].values())
    return item


def apply(store, tree, op):
    action = op['op']
    if action == 'queue-enable':
        require(tree['schema'] == 'operator-harness/search@1', 'queue already enabled')
        require(all(settled(run) for run in tree['runs'].values()), 'enable queue at a safe boundary')
        for run_id in tree['runs']:
            control.reconciled(tree, run_id)
        require(type(op.get('max_parallel')) is int and op['max_parallel'] > 0, 'invalid max_parallel')
        tree['schema'] = SCHEMA
        tree['scheduling'] = {'max_parallel': op['max_parallel'], 'authority': store.reference(op['authority']),
                              'items': {}, 'leases': {}, 'groups': {}}
        return
    require(tree['schema'] == SCHEMA, 'enable queue first')
    scheduling, items = tree['scheduling'], tree['scheduling']['items']
    key = identifier(op['id'])
    if action == 'enqueue':
        require(key not in items, 'queue item already exists')
        item = proposal(store, tree, op['spec'])
        for old in items.values():
            fields = set(item) - {'reason', 'priority', 'supersedes'}
            require(old['state'] != 'queued' or any(old.get(k) != item[k] for k in fields), 'duplicate queued work')
        items[key] = dict(item, revision=1, state='queued')
    elif action == 'queue-revise':
        old = items[key]
        require(old['state'] == 'queued' and old['revision'] == op['revision'], 'stale or admitted queue item')
        items[key] = dict(proposal(store, tree, op['spec']), revision=old['revision'] + 1, state='queued',
                          deferrals=old.get('deferrals', []))
    elif action in ('queue-cancel', 'queue-defer'):
        item = items[key]
        require(item['state'] == 'queued' and item['revision'] == op['revision'], 'stale or admitted queue item')
        text(op.get('reason'), 'decision reason')
        item['revision'] += 1
        if action == 'queue-cancel':
            item.update(state='cancelled', reason=op['reason'])
        else:
            item.setdefault('deferrals', []).append({'reason': op['reason'], 'event_id': op['event_id']})
    elif action == 'comparison-group':
        require(key not in scheduling['groups'], 'comparison group already exists')
        require(isinstance(op['members'], list) and len(op['members']) >= 2 and
                len(op['members']) == len(set(op['members'])), 'group needs distinct members')
        require(all(items[k]['state'] == 'queued' for k in op['members']), 'group must be fixed before admission')
        require(all(items[k]['stage'] == 'develop' for k in op['members']), 'comparison members must be develop')
        require(len({items[k]['candidate_id'] for k in op['members']}) == 1, 'comparison parent candidates differ')
        refs = immutable_refs(store, op['inputs'].values())
        require(set(op['inputs']) == {'requirements', 'baseline', 'cases', 'protocol', 'environment'},
                'comparison group needs complete frozen comparison inputs')
        scheduling['groups'][key] = {'members': list(op['members']), 'inputs': deepcopy(op['inputs']),
                                   'dependencies': refs, 'candidate_id': items[op['members'][0]]['candidate_id']}
    elif action == 'queue-admit':
        item = items[key]
        require(item['revision'] == op['revision'], 'stale queue revision')
        run_id = identifier(op['run_id'])
        require(run_id not in tree['runs'], 'run already registered')
        request_id = identifier(op['request_id'])
        require(all(r.get('dispatch_request_id') != request_id for r in tree['runs'].values()), 'dispatch request already used')
        text(op.get('reason'), 'admission reason')
        refs, resources = checks(store, tree, item, run_id)
        task = store.reference(op['task'])
        require(store.local(task['path']) == store.records / 'runs' / run_id / 'TASK.md', 'TASK must use own run directory')
        store.markdown(task, run_id=run_id, node=item['stage'], search_node_id=item['search_node_id'],
                       queue_item_id=key, operation=item['operation'])
        for field in ('plan_id', 'mechanism_id'):
            if item.get(field):
                store.markdown(task, **{field: item[field]})
        run = {'search_node_id': item['search_node_id'], 'stage': item['stage'],
               'candidate_id': item['candidate_id'], 'task': task, 'result': None,
               'queue_item_id': key, 'dispatch_request_id': request_id, 'repo_ids': item['repo_ids']}
        if item.get('analysis_run'):
            store.markdown(task, analysis_run=item['analysis_run'])
            run['analysis_run'] = item['analysis_run']
        if item['stage'] == 'develop':
            store.develop_scope(task)
        store.admit(run, False, tree)
        refs.update(stable_refs(store, run.get('evaluation_inputs', {}).values()))
        if run.get('initial_design'):
            refs.update(stable_refs(store, [run['initial_design']]))
        for resource in resources:
            if resource.get('identity_evidence'):
                refs[resource['identity_evidence']['path']] = resource['identity_evidence']
        for group in scheduling['groups'].values():
            if key in group['members']:
                refs.update(immutable_refs(store, group['inputs'].values()))
                require(run.get('evaluation_plan'), 'comparison member needs evaluation plan')
                plan, deps = evidence.plan(store, run['evaluation_plan'])
                for name in ('requirements', 'cases', 'protocol', 'environment'):
                    require(plan['inputs'][name] == group['inputs'].get(name), 'comparison inputs differ')
                require(group['inputs'].get('baseline') in deps.values(), 'comparison baseline missing')
                identity = evidence.comparison_identity(plan)
                require(group.get('comparison', identity) == identity, 'comparison plans differ')
                group['comparison'] = identity
        run['queue_inputs'] = refs
        tree['runs'][run_id] = run
        generation = 1 + max((lease['generation'] for lease in scheduling['leases'].values()), default=0)
        scheduling['leases'][run_id] = {'generation': generation, 'released': False, 'resources': resources}
        item.update(state='admitted', run_id=run_id, admission_reason=op['reason'])
    elif action == 'release-workspace':
        run, lease = tree['runs'][key], scheduling['leases'][key]
        require(settled(run), 'run has not settled')
        require(not lease['released'] and lease['generation'] == op['generation'], 'stale or released lease')
        control.reconciled(tree, key)
        ref = store.reference(op['evidence'])
        record = json.loads(store.local(ref['path']).read_text())
        require(record.get('schema') == 'operator-harness/workspace-release@1' and
                record.get('project_id') == store.project_id and record.get('run_id') == key and
                record.get('generation') == lease['generation'], 'release identity mismatch')
        require(record.get('writers_stopped') is True and record.get('jobs') == [] and
                record.get('unsettled_resources') == [], 'unconfirmed writers/jobs/resources')
        require(isinstance(record.get('evidence'), list) and record['evidence'], 'release needs observation evidence')
        for item in record['evidence']:
            store.verify_ref(item)
        if run.get('closure'):
            store.validate_closure(key, run)
        else:
            store.verify_run(tree, key)
        for cid, candidate in tree['candidates'].items():
            from records import metadata
            if metadata(store.local(candidate['manifest']['path']).read_text()).get('producer_run') == key:
                candidate_inputs(store, tree, cid)
        lease.update(released=True, evidence=ref)
    else:
        raise Invalid('unknown queue operation')


def verify_run(store, tree, run_id):
    run = tree['runs'][run_id]
    for ref in run.get('queue_inputs', {}).values():
        store.verify_ref(ref)


def before_maintenance(tree):
    require(not active_leases(tree), 'release all queue writers before maintenance')
    require(all(settled(run) for run in tree['runs'].values()), 'settle all runs before maintenance')
