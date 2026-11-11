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
    with workspace.edit(project, owner):
        if directory.exists():
            saved = directory / 'request.json'
            projectctl.require(saved.exists() and projectctl.read(saved) == spec, 'request conflict or incomplete intent')
            prior = status(directory)
        if prior is not None:
            # Reconcile after the workspace lock is released; never execute twice.
            pass
        else:
            prepare_request(workspace, project, spec, directory, fingerprint, local)
    if prior is not None:
        if prior['status'] in ('exited', 'timed_out', 'interrupted', 'failed_to_start', 'not_submitted'):
            ledger(store, spec, directory, control_state(prior, directory), prior)
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
        ledger(store, spec, directory, 'not_submitted', receipt)
        return receipt
    return launch(store, spec, directory, argv, timeout, env, cwd, state, fingerprint)


def prepare_request(workspace, project, spec, directory, fingerprint, local):
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
    if outcome in ('not_submitted', 'failed_to_start'):
        return 'not_submitted'
    if outcome in ('timed_out', 'interrupted'):
        state = projectctl.read(directory / 'state.json') if (directory / 'state.json').exists() else {}
        child = state.get('pid')
        if child and state.get('process_start') and start_identity(child) == state['process_start']:
            return 'unknown'
        return 'cancelled'
    return 'unknown'


def ledger(store, spec, directory, state, receipt=None):
    if not store.path.exists():
        return
    tree = store.load()
    if 'control' not in tree:
        return
    request = spec['request_id']
    reservation_id = 'job-' + request
    if receipt is not None and reservation_id not in tree['control']['reservations']:
        return
    budget = spec.get('reservation')
    projectctl.require(budget is not None, 'controlled execution requires reservation pool/amount')
    pool = tree['control']['budgets'][budget['pool']]
    projectctl.require(pool['unit'] in ('seconds', 'executions'), 'runner supports seconds/executions budget units')
    def apply(action, **fields):
        store.apply(dict(event_id=request + '-' + action + ('-' + fields['status'] if 'status' in fields else ''), op=action, **fields))
    if reservation_id not in tree['control']['reservations']:
        if receipt is not None:
            return
        apply('control-reserve', id=reservation_id, run_id=spec['run_id'], **budget)
    job = store.load()['control']['jobs'].get(request)
    if (job is None or job['status'] != state) and not (job is None and state == 'not_submitted'):
        path = directory / ('observation-' + state + '.json')
        if not path.exists():
            projectctl.atomic(path, {'schema': 'operator-harness/job-observation@1',
                'project_id': store.project_id, 'run_id': spec['run_id'], 'request_id': request,
                'status': state, 'observed_at': datetime.now(timezone.utc).isoformat(),
                'source': 'runctl local process supervisor',
                'process_receipt': store.reference(str((directory / 'process-receipt.json').relative_to(store.root))) if receipt else None})
        apply('control-job', id=request, status=state, reservation_id=reservation_id,
              evidence=str(path.relative_to(store.root)))
    if receipt and state in ('exited', 'cancelled', 'not_submitted') and not store.load()['control']['reservations'][reservation_id]['settled']:
        used = 0 if state == 'not_submitted' else (receipt['elapsed_seconds'] if pool['unit'] == 'seconds' else 1)
        apply('control-settle', id=reservation_id, used=used,
              evidence=str((directory / 'process-receipt.json').relative_to(store.root)))


def launch(store, spec, directory, argv, timeout, env, cwd, state, fingerprint):
    project, run, request = spec['project_id'], spec['run_id'], spec['request_id']
    began = time.monotonic()
    process = None
    code = None
    failure = None
    outcome = 'exited'
    try:
        with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **env}, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            state.update(status='running', pid=process.pid, process_start=start_identity(process.pid))
            projectctl.atomic(directory / 'state.json', state)
            ledger(store, spec, directory, 'running')
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                outcome = 'timed_out'
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The process may exit between timeout detection and cancellation.
                code = process.wait()
    except BaseException as exc:
        outcome = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed_to_start'
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if process is not None:
            # Foreground contract: also terminate surviving members of this process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
    receipt = {'schema': 'operator-harness/process-receipt@1', 'project_id': project, 'run_id': run,
               'request_id': request, 'request_sha256': fingerprint, 'status': outcome,
               'exit_codes': {'process': code}, 'elapsed_seconds': time.monotonic() - began,
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
