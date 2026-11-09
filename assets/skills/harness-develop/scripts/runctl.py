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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'harness-new-project/scripts'))
import projectctl


def start_identity(pid):
    try:
        # Linux starttime, after the parenthesized comm field (which may contain spaces).
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def location(workspace, project, run, request):
    for value in (run, request):
        projectctl.require(isinstance(value, str) and value and all(c.isalnum() or c in '._-' for c in value)
                           and value not in ('.', '..'), 'invalid run/request ID')
    return workspace.local(project, f'runs/{run}/jobs/{request}')


def status(directory):
    receipt = directory / 'process-receipt.json'
    if receipt.exists():
        return projectctl.read(receipt)
    state_path = directory / 'state.json'
    state = projectctl.read(state_path) if state_path.exists() else {}
    pid = state.get('supervisor_pid')
    alive = bool(pid and state.get('supervisor_start') and start_identity(pid) == state['supervisor_start'])
    return {**state, 'status': 'running' if alive else 'unknown',
            'message': 'No terminal receipt; query/recover the original process, do not resubmit.'}


def execute(workspace, project, owner, spec):
    projectctl.require(sys.platform == 'linux', 'local runner currently supports Linux only')
    projectctl.require(spec.get('schema') == 'operator-harness/process-request@1'
                       and spec.get('project_id') == project, 'unsupported request/identity')
    run, request = spec['run_id'], spec['request_id']
    directory = location(workspace, project, run, request)
    encoded = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    # Persistent request directory is the at-most-once barrier, including the Popen gap.
    with workspace.edit(project, owner):
        if directory.exists():
            saved = directory / 'request.json'
            projectctl.require(saved.exists() and projectctl.read(saved) == spec, 'request conflict or incomplete intent')
            return status(directory)
        argv = spec.get('argv')
        projectctl.require(isinstance(argv, list) and bool(argv) and
                           all(isinstance(v, str) and '\0' not in v for v in argv), 'invalid argv')
        timeout = spec.get('timeout_seconds')
        projectctl.require(type(timeout) in (int, float) and 0 < timeout <= 86400, 'invalid bounded timeout')
        cwd = workspace.local(project, spec['cwd'])
        projectctl.require(cwd.is_dir(), 'working directory missing')
        inputs = spec.get('inputs')
        projectctl.require(isinstance(inputs, list) and bool(inputs), 'declare hashed execution inputs')
        for ref in inputs:
            path = workspace.local(project, ref['path'])
            projectctl.require(projectctl.sha(path) == ref['sha256'], 'execution input changed')
        env = spec.get('env', {})
        projectctl.require(isinstance(env, dict) and all(isinstance(k, str) and k and '=' not in k and '\0' not in k
                           and isinstance(v, str) and '\0' not in v for k, v in env.items()), 'invalid environment')
        directory.mkdir(parents=True)
        projectctl.atomic(directory / 'request.json', spec)
        state = {'schema': 'operator-harness/process-state@1', 'project_id': project, 'run_id': run,
                 'request_id': request, 'request_sha256': fingerprint, 'status': 'intent',
                 'supervisor_pid': os.getpid(), 'supervisor_start': start_identity(os.getpid())}
        projectctl.atomic(directory / 'state.json', state)
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
