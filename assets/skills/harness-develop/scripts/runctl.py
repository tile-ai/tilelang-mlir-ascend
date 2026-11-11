#!/usr/bin/env python3
"""Durable local foreground process runner. Not a remote/NPU adapter or scheduler."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'harness-new-project/scripts'))
import projectctl
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'harness-resume/scripts'))
import searchctl


def start_identity(pid):
    try:
        # Linux starttime, after the parenthesized comm field (which may contain spaces).
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def location(workspace, project, run, request):
    for value in (run, request):
        projectctl.require(isinstance(value, str) and value and all(c.isalnum() or c in '._-' for c in value)
                           and value not in ('.', '..'), 'invalid run/request ID')
    return workspace.local(project, f'runs/{run}/jobs/{request}')


def status(directory):
    if not directory.exists():
        return {'status': 'not_found', 'message': 'No submission intent exists.'}
    receipt = directory / 'process-receipt.json'
    if receipt.exists():
        return projectctl.read(receipt)
    state_path = directory / 'state.json'
    state = projectctl.read(state_path) if state_path.exists() else {}
    pid = state.get('supervisor_pid')
    alive = bool(pid and state.get('supervisor_start') and start_identity(pid) == state['supervisor_start'])
    child = state.get('pid')
    child_alive = bool(child and state.get('process_start') and start_identity(child) == state['process_start'])
    return {**state, 'status': 'running' if alive or child_alive else 'unknown',
            'supervisor_alive': alive, 'process_alive': child_alive, 'orphaned': child_alive and not alive,
            'message': 'No terminal receipt; query/recover the original process, do not resubmit.'}


def execute(workspace, project, owner, spec):
    projectctl.require(sys.platform == 'linux', 'local runner currently supports Linux only')
    projectctl.require(spec.get('schema') in ('operator-harness/process-request@1', 'operator-harness/process-request@2')
                       and spec.get('project_id') == project, 'unsupported request/identity')
    run, request = spec['run_id'], spec['request_id']
    directory = location(workspace, project, run, request)
    encoded = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    store = searchctl.Store(workspace.root, project, owner)
    local = store.local if spec['schema'].endswith('@2') else lambda path: workspace.local(project, path)
    prior = None
    # Persistent request directory is the at-most-once barrier, including the Popen gap.
    with workspace.edit(project, owner, states=('active', 'freezing')) as project_state:
        if directory.exists():
            saved = directory / 'request.json'
            projectctl.require(saved.exists() and projectctl.read(saved) == spec, 'request conflict or incomplete intent')
            prior = status(directory)
        if prior is not None:
            # Reconcile after the workspace lock is released; never execute twice.
            pass
        else:
            projectctl.require(project_state['status'] == 'active', 'new process request requires an active project')
            prepare_request(workspace, project, spec, directory, fingerprint, local)
    if prior is not None:
        if prior['status'] in ('exited', 'timed_out', 'interrupted', 'failed_to_start', 'not_submitted'):
            try:
                ledger(store, spec, directory, control_state(prior, directory), prior)
            except (ValueError, OSError, KeyError):
                if prior['status'] != 'not_submitted' or not prior.get('ledger_error'):
                    raise
        return prior
    argv, timeout, env, cwd = spec['argv'], spec['timeout_seconds'], spec.get('env', {}), local(spec['cwd'])
    state = projectctl.read(directory / 'state.json')
    try:
        ledger(store, spec, directory, 'intent')
    except (ValueError, OSError, KeyError) as exc:
        receipt = {'schema': 'operator-harness/process-receipt@1', 'project_id': project, 'run_id': run,
                   'request_id': request, 'request_sha256': fingerprint, 'status': 'not_submitted',
                   'exit_codes': {'process': None}, 'elapsed_seconds': 0,
                   'error': str(exc), 'validation_scope': 'local_process_only', 'device_validation': 'not_evaluated', 'logs': {}}
        projectctl.atomic(directory / 'process-receipt.json', receipt)
        try:
            ledger(store, spec, directory, 'not_submitted', receipt)
        except (ValueError, OSError, KeyError) as ledger_exc:
            # A conflicting request belongs to its original run. Never settle it
            # while recording this rejected submission.
            receipt['ledger_error'] = str(ledger_exc)
            projectctl.atomic(directory / 'process-receipt.json', receipt)
        return receipt
    return launch(store, spec, directory, argv, timeout, env, cwd, state, fingerprint)


def prepare_request(workspace, project, spec, directory, fingerprint, local):
    projectctl.require(type(spec.get('defer_settlement', False)) is bool, 'defer_settlement must be boolean')
    argv = spec.get('argv')
    projectctl.require(isinstance(argv, list) and bool(argv) and
                       all(isinstance(v, str) and '\0' not in v for v in argv), 'invalid argv')
    timeout = spec.get('timeout_seconds')
    projectctl.require(type(timeout) in (int, float) and 0 < timeout <= 86400, 'invalid bounded timeout')
    cwd = local(spec['cwd'])
    projectctl.require(cwd.is_dir(), 'working directory missing')
    inputs = spec.get('inputs')
    projectctl.require(isinstance(inputs, list) and bool(inputs), 'declare hashed execution inputs')
    for ref in inputs:
        path = local(ref['path'])
        projectctl.require(projectctl.sha(path) == ref['sha256'], 'execution input changed')
    env = spec.get('env', {})
    projectctl.require(isinstance(env, dict) and all(isinstance(k, str) and k and '=' not in k and '\0' not in k
                       and isinstance(v, str) and '\0' not in v for k, v in env.items()), 'invalid environment')
    directory.mkdir(parents=True)
    projectctl.atomic(directory / 'request.json', spec)
    state = {'schema': 'operator-harness/process-state@1', 'project_id': project, 'run_id': spec['run_id'],
             'request_id': spec['request_id'], 'request_sha256': fingerprint, 'status': 'intent',
             'supervisor_pid': os.getpid(), 'supervisor_start': start_identity(os.getpid())}
    projectctl.atomic(directory / 'state.json', state)

def control_state(receipt, directory):
    outcome = receipt['status']
    if outcome == 'exited':
        return 'exited'
    if outcome == 'not_submitted':
        return 'not_submitted'
    if outcome in ('timed_out', 'interrupted', 'failed_to_start'):
        state = projectctl.read(directory / 'state.json') if (directory / 'state.json').exists() else {}
        if outcome == 'failed_to_start' and not receipt.get('process_started') and not state.get('pid'):
            return 'not_submitted'
        child = receipt.get('process_pid') or state.get('pid')
        started = receipt.get('process_start') or state.get('process_start')
        if receipt.get('process_started') and receipt.get('process_stopped') is False and not (child and started):
            return 'unknown'
        if child and started and start_identity(child) == started:
            return 'unknown'
        return 'cancelled'
    return 'unknown'


def ledger(store, spec, directory, state, receipt=None):
    if not store.path.exists():
        return
    # Check identity, reserve and record intent under one lock. A second run
    # cannot observe a partial reservation and borrow it before Popen.
    with store.write(closing=state != 'intent'):
        tree = store.load()
        if 'control' not in tree:
            return
        request = spec['request_id']
        reservation_id = 'job-' + request
        c = tree['control']
        reservation = c['reservations'].get(reservation_id)
        job = c['jobs'].get(request)
        if receipt is not None and reservation is None:
            return
        budget = spec.get('reservation')
        projectctl.require(isinstance(budget, dict) and set(budget) == {'pool', 'amount'},
                           'controlled execution requires reservation pool/amount')
        pool = c['budgets'][budget['pool']]
        projectctl.require(pool['unit'] in ('seconds', 'executions'), 'runner supports seconds/executions budget units')
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if reservation is not None:
            projectctl.require(reservation['run_id'] == spec['run_id'] and
                               reservation['pool'] == budget['pool'] and reservation['amount'] == budget['amount'],
                               'request reservation identity/budget conflict')
        if job is not None:
            projectctl.require(job['run_id'] == spec['run_id'] and job['reservation_id'] == reservation_id,
                               'request job identity conflict')
        for record in (reservation, job):
            if record is None:
                continue
            if record.get('request_sha256'):
                projectctl.require(record['request_sha256'] == fingerprint, 'request specification conflict')
            else:
                # Existing runner records predate fingerprint fields. Their
                # persisted original request is still required for safe recovery.
                saved = directory / 'request.json'
                projectctl.require(saved.exists() and projectctl.read(saved) == spec,
                                   'legacy request specification cannot be verified')
        if receipt is not None:
            projectctl.require(all(receipt.get(key) == spec[key] for key in ('project_id', 'run_id', 'request_id'))
                               and receipt.get('request_sha256') == fingerprint, 'receipt request identity conflict')
        if state == 'intent':
            projectctl.require(not c['paused'], 'control is paused')
            run = tree['runs'][spec['run_id']]
            projectctl.require(not run.get('result') and not run.get('closure'), 'run already settled')
            projectctl.require(reservation is None or not reservation['settled'], 'reservation already settled')
            projectctl.require(job is None or job['status'] == 'intent', 'request already submitted')

        def apply(action, **fields):
            operation = dict(event_id=request + '-' + action + ('-' + fields['status'] if 'status' in fields else ''),
                             op=action, **fields)
            searchctl.identifier(operation['event_id'])
            for event in tree['events']:
                if event['operation']['event_id'] == operation['event_id']:
                    projectctl.require(event['operation'] == operation, 'event ID reused with different content')
                    return
            store.mutate(tree, operation)
            tree['events'].append({'at': searchctl.now(), 'operation': operation})

        if reservation is None:
            apply('control-reserve', id=reservation_id, run_id=spec['run_id'], **budget,
                  request_sha256=fingerprint)
            reservation = c['reservations'][reservation_id]
            reservation['request_sha256'] = fingerprint
        if (job is None or job['status'] != state) and not (job is None and state == 'not_submitted'):
            path = directory / ('observation-' + state + '.json')
            if not path.exists():
                projectctl.atomic(path, {'schema': 'operator-harness/job-observation@1',
                    'project_id': store.project_id, 'run_id': spec['run_id'], 'request_id': request,
                    'status': state, 'observed_at': datetime.now(timezone.utc).isoformat(),
                    'source': 'runctl local process supervisor',
                    'process_receipt': store.reference(str((directory / 'process-receipt.json').relative_to(store.root))) if receipt else None})
            apply('control-job', id=request, status=state, reservation_id=reservation_id,
                  evidence=str(path.relative_to(store.root)), request_sha256=fingerprint)
            c['jobs'][request]['request_sha256'] = fingerprint
        if receipt and state in ('exited', 'cancelled', 'not_submitted') and not reservation['settled']:
            if spec.get('defer_settlement') and state != 'not_submitted':
                reservation['settlement_required'] = 'final resource release and actual usage evidence'
            elif pool['unit'] == 'seconds' and receipt.get('process_stopped') is False and state != 'not_submitted':
                # The receipt ended before the process was confirmed stopped.
                # Its elapsed time excludes the unknown interval; retain the
                # reservation until the host supplies actual usage evidence.
                reservation['settlement_required'] = 'host usage evidence for unobserved process duration'
            else:
                used = 0 if state == 'not_submitted' else (receipt['elapsed_seconds'] if pool['unit'] == 'seconds' else 1)
                apply('control-settle', id=reservation_id, used=used,
                      evidence=str((directory / 'process-receipt.json').relative_to(store.root)))
        store.validate(tree)
        searchctl.atomic(store.path, tree)


def launch(store, spec, directory, argv, timeout, env, cwd, state, fingerprint):
    project, run, request = spec['project_id'], spec['run_id'], spec['request_id']
    began = time.monotonic()
    process = None
    code = None
    failure = None
    outcome = 'exited'
    stopped = False
    try:
        with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
            with store.workspace.edit(project, store.owner):
                process = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **env}, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                state.update(status='running', pid=process.pid, process_start=start_identity(process.pid))
                projectctl.atomic(directory / 'state.json', state)
            ledger(store, spec, directory, 'running')
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                outcome = 'timed_out'
    except BaseException as exc:
        outcome = 'interrupted' if process is not None or isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed_to_start'
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if process is not None:
            # Foreground contract: also terminate surviving members of this process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                failure = (failure + '; ' if failure else '') + f'process cleanup: {exc}'
                outcome = 'interrupted'
            try:
                code = process.wait(timeout=5)
                stopped = True
            except subprocess.TimeoutExpired:
                outcome = 'interrupted'
                failure = (failure + '; ' if failure else '') + 'process cleanup is unconfirmed'
    receipt = {'schema': 'operator-harness/process-receipt@1', 'project_id': project, 'run_id': run,
               'request_id': request, 'request_sha256': fingerprint, 'status': outcome,
               'exit_codes': {'process': code}, 'elapsed_seconds': time.monotonic() - began,
               'process_started': process is not None, 'process_stopped': stopped,
               'process_pid': process.pid if process is not None else None,
               'process_start': state.get('process_start'),
               'error': failure, 'validation_scope': 'local_process_only',
               'device_validation': 'not_evaluated',
               'logs': {name: projectctl.sha(directory / name) for name in ('stdout.log', 'stderr.log')}}
    projectctl.atomic(directory / 'process-receipt.json', receipt)
    ledger(store, spec, directory, control_state(receipt, directory), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--owner')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('run').add_argument('request', type=Path)
    query = commands.add_parser('status')
    query.add_argument('--run', required=True)
    query.add_argument('--request', required=True)
    args = parser.parse_args()
    try:
        workspace = projectctl.Workspace(args.workspace)
        workspace.load(args.project)
        if args.command == 'status':
            result = status(location(workspace, args.project, args.run, args.request))
        else:
            result = execute(workspace, args.project, args.owner, projectctl.read(args.request))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == 'status':
            return 0
        if result['status'] != 'exited':
            return 124 if result['status'] == 'timed_out' else 1
        code = result['exit_codes']['process']
        return code if code >= 0 else 128 - code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'runctl: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
