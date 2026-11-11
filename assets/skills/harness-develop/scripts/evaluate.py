#!/usr/bin/env python3
"""Execute a frozen Python evaluation adapter and emit consumable run evidence."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import emission
import runctl
from runctl import projectctl, searchctl
from records import render


def load_module(store, ref, name):
    store.verify_ref(ref)
    spec = importlib.util.spec_from_file_location(name, store.local(ref['path']))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serializable(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('output must be JSON-compatible or a tensor')


@contextmanager
def prepared_case(adapter, context, case):
    """Release each case on success and failure; cleanup failures invalidate the run."""
    state = adapter.prepare(context, case)
    try:
        yield state
    finally:
        try:
            adapter.synchronize(state)
        finally:
            teardown = getattr(adapter, 'teardown', None)
            if callable(teardown):
                teardown(state)


def accuracy_case(store, adapter, reference, comparator, context, case, output):
    # A separate frame releases state AND actual/reference tensors before the next prepare.
    with prepared_case(adapter, context, case) as state:
        expected = reference.run(state)
        actual = adapter.run(state)
        adapter.synchronize(state)
        raw = emission.save(output / 'outputs.json', {'actual': serializable(actual), 'expected': serializable(expected)})
        outputs = comparator.compare(actual, expected, case)
        searchctl.require(isinstance(outputs, dict) and set(outputs) == set(case['outputs']) and
                          all(type(v) is bool for v in outputs.values()), 'comparator must return all named outputs')
        common = {key: context[key] for key in ('project_id', 'run_id', 'candidate_id', 'case_id', 'request_id', 'artifacts', 'evaluation_plan')}
        return dict(common, schema='operator-harness/accuracy@1',
                    raw_files=[emission.reference(store.root, raw)], outputs=outputs)


def collect_case(store, adapter, state, context, case, output):
    capture = adapter.diagnostics(state, case, output, context)
    verifier = getattr(adapter, 'verify_diagnostics', None)
    if callable(verifier):
        verifier(capture, case)
    ref = capture['evidence']
    store.verify_ref(ref)
    report = json.loads(store.local(ref['path']).read_text())
    searchctl.require(report['schema'] == 'operator-harness/diagnostic-evidence@1' and
                      report['candidate'] == context['candidate'], 'diagnostic collector identity mismatch')
    return ref


def measured_case(store, adapter, context, case, output, data):
    result = {}
    with prepared_case(adapter, context, case) as state:
        if data['performance_kind'] != 'none':
            if data['execution']['backend'] == 'torch_npu':
                from torch_npu_measure import measure
                options = adapter.measurement(state, case, output / 'measure')
                report = measure(**options, evidence_context=context)
                searchctl.require(report.get('status') == 'pending_guard_receipt', 'unexpected device measurement status')
                result['performance'] = report['evidence']
            else:
                samples = []
                for _ in range(case['min_samples']):
                    adapter.synchronize(state)
                    began = time.perf_counter_ns()
                    actual = adapter.run(state)
                    adapter.synchronize(state)
                    samples.append((time.perf_counter_ns() - began) / 1e6)
                    del actual
                raw = emission.save(output / 'samples.json', {'metric': 'cpu_wall_ms', 'samples_ms': samples})
                common = {key: context[key] for key in ('project_id', 'run_id', 'candidate_id', 'case_id', 'request_id', 'artifacts', 'evaluation_plan')}
                result['performance'] = dict(common, schema='operator-harness/performance@1',
                    metric='cpu_wall_ms', raw_files=[emission.reference(store.root, raw)], samples_ms=samples)
        if context['diagnostics_required']:
            result['diagnostic'] = collect_case(store, adapter, state, context, case, output / 'diagnostics')
    return result


def worker(store, context):
    """Runs only in the bounded child process; comparator and cleanup errors propagate."""
    data, _ = searchctl.evidence.plan(store, context['evaluation_plan'])
    adapter = load_module(store, data['execution']['adapter'], 'harness_adapter')
    root = store.local(context['output'])
    rows, diagnostics = [], []
    if context.get('operation') == 'collect-only':
        # measured_case handles one state lifetime even when performance is disabled.
        collection_plan = dict(data, performance_kind='none')
        for index, case in enumerate(data['cases']):
            ctx = dict(context, workspace_root=str(store.root), case_id=case['id'], diagnostics_required=True)
            result = measured_case(store, adapter, ctx, case, root / f'case-{index}', collection_plan)
            diagnostics.append(result['diagnostic'])
    else:
        reference = load_module(store, data['inputs']['reference'], 'harness_reference')
        comparator = load_module(store, data['inputs']['comparator'], 'harness_comparator')
        # Finish correctness for ALL cases before any timing, without retaining case buffers.
        for index, case in enumerate(data['cases']):
            ctx = dict(context, workspace_root=str(store.root), case_id=case['id'])
            report = accuracy_case(store, adapter, reference, comparator, ctx, case, root / f'case-{index}')
            rows.append({'id': case['id'], 'accuracy': report})
        if not all(all(row['accuracy']['outputs'].values()) for row in rows):
            emission.save(root / 'worker-result.json', {'status': 'accuracy_failed', 'cases': rows, 'diagnostics': []})
            return
        if data['performance_kind'] != 'none' or context['diagnostics_required']:
            for index, (row, case) in enumerate(zip(rows, data['cases'])):
                ctx = dict(context, workspace_root=str(store.root), case_id=case['id'])
                result = measured_case(store, adapter, ctx, case, root / f'case-{index}', data)
                if 'performance' in result:
                    row['performance'] = result['performance']
                if 'diagnostic' in result:
                    diagnostics.append(result['diagnostic'])
    emission.save(root / 'worker-result.json', {
        'status': 'complete', 'cases': rows, 'diagnostics': diagnostics,
        'diagnostics_verified': bool(diagnostics and callable(getattr(adapter, 'verify_diagnostics', None))),
    })


def candidate(store, run_id, candidate_id, source, build, launch, parent_id):
    directory = store.records / 'runs' / run_id / 'candidates' / searchctl.identifier(candidate_id)
    artifacts = {key: store.reference(path) for key, path in (('source', source), ('build', build), ('launch', launch))}
    path = directory / 'CANDIDATE.md'
    with store.write():
        tree = store.load()
        run = tree['runs'][run_id]
        if run.get('queue_item_id'):
            searchctl.require(not searchctl.settled(run) and
                              not tree['scheduling']['leases'][run_id]['released'],
                              'candidate writer run is settled or released')
            artifacts = {key: searchctl.scheduler.frozen_blob(store, ref) for key, ref in artifacts.items()}
        record = {'contract': 'operator-harness/candidate@2', 'project_id': store.project_id,
                  'search_node_id': run['search_node_id'], 'candidate_id': candidate_id,
                  'producer_run': run_id, 'parent_candidate': parent_id or 'not_applicable'}
        for key, ref in artifacts.items():
            record[key] = ref['path']
            record[key + '_sha256'] = ref['sha256']
        text = render('Candidate', record)
        if path.exists():
            searchctl.require(path.read_text() == text, 'candidate_id already exists with different artifacts; use a new candidate_id')
        else:
            searchctl.require(candidate_id not in tree['candidates'], 'candidate_id already registered; use a new candidate_id')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    store.apply({'event_id': 'candidate-' + candidate_id, 'op': 'candidate', 'id': candidate_id,
                 'search_node_id': run['search_node_id'], 'parent_candidate': parent_id,
                 'manifest': str(path.relative_to(store.root))})
    return {'candidate': str(path.relative_to(store.root))}


def evaluate(store, run_id, candidate_path, request_id, reservation=None, device_receipt_path=None):
    tree = store.load()
    store.verify_run(tree, run_id)
    run = tree['runs'][run_id]
    data = json.loads(store.local(run['evaluation_plan']['path']).read_text())
    dependencies = dict(run.get('evaluation_inputs') or searchctl.evidence.plan(store, run['evaluation_plan'])[1])
    searchctl.require(data.get('execution'), 'plan needs a frozen execution adapter')
    searchctl.require(not device_receipt_path or data['execution']['backend'] == 'torch_npu',
                      'device receipt is only valid for torch_npu execution')
    candidate_ref = store.reference(candidate_path)
    store.markdown(candidate_ref, search_node_id=run['search_node_id'])
    artifacts = {key: store.reference(store.value(candidate_ref, key)) for key in ('source', 'build', 'launch')}
    for key, ref in artifacts.items():
        searchctl.require(ref['sha256'] == store.value(candidate_ref, key + '_sha256'), 'candidate artifact changed')
        searchctl.evidence.freeze(store, ref, dependencies)
    candidate_id = store.value(candidate_ref, 'candidate_id')
    operation = store.value(run['task'], 'operation')
    searchctl.require(candidate_id in tree['candidates'] and tree['candidates'][candidate_id]['manifest'] == candidate_ref,
                      'register the exact candidate before execution')
    if operation in ('verify-only', 'collect-only'):
        searchctl.require(candidate_ref == run['input_candidate'], 'read-only execution changed candidate')
    elif operation == 'implement-and-validate':
        searchctl.require(candidate_id != run['candidate_id'], 'implementation must produce a new candidate')
        store.parent(candidate_ref, run['candidate_id'], run['input_candidate'])
    output = store.records / 'runs' / run_id / 'evaluations' / searchctl.identifier(request_id)
    context = {'project_id': store.project_id, 'run_id': run_id, 'candidate_id': candidate_id,
               'candidate': candidate_ref, 'request_id': request_id, 'artifacts': artifacts,
               'evaluation_plan': run['evaluation_plan'], 'output': str(output.relative_to(store.root)),
               'diagnostics_required': store.value(run['task'], 'diagnostics_required') == 'true',
               'operation': store.value(run['task'], 'operation')}
    context_path = output / 'context.json'
    prior_request = runctl.location(store.workspace, store.project_id, run_id, request_id) / 'request.json'
    recovering = context_path.is_file() and prior_request.is_file()
    with store.write(closing=recovering):
        if context_path.exists():
            searchctl.require(json.loads(context_path.read_text()) == context, 'evaluation request conflict')
        else:
            emission.save(context_path, context)
    spec = {'schema': 'operator-harness/process-request@2', 'project_id': store.project_id,
            'run_id': run_id, 'request_id': request_id, 'cwd': '.',
            'timeout_seconds': data['execution']['timeout_seconds'],
            'argv': [sys.executable, str(Path(__file__).resolve()), '--workspace', str(store.root),
                     '--project', store.project_id, 'worker', str(context_path)],
            'inputs': list(dependencies.values()) + [candidate_ref, store.reference(str(context_path.relative_to(store.root)))]}
    if reservation:
        spec['reservation'] = reservation
    if data['execution']['backend'] == 'torch_npu':
        spec['defer_settlement'] = True
    process = runctl.execute(store.workspace, store.project_id, store.owner, spec)
    if process['status'] != 'exited' or process['exit_codes']['process'] != 0:
        return {'status': process['status'], 'process': process}
    produced = json.loads((output / 'worker-result.json').read_text())
    accuracy = 'passed' if produced['status'] == 'complete' else 'failed'
    worker_result = store.reference(str((output / 'worker-result.json').relative_to(store.root)))
    final_device = None
    if data['execution']['backend'] == 'torch_npu' and accuracy == 'passed':
        finalization = output / 'device-finalization.json'
        saved = json.loads(finalization.read_text()) if finalization.exists() else None
        final_device = store.reference(device_receipt_path) if device_receipt_path else saved
        if final_device is None:
            return {'status': 'pending_guard_receipt', 'context': str(context_path.relative_to(store.root)),
                    'worker_result': worker_result, 'process': process,
                    'message': 'Collect the original guard receipt, then repeat this request with --device-receipt; the worker will not run again.'}
        searchctl.require(saved is None or saved == final_device, 'device finalization already binds another receipt')
        device = searchctl.evidence.device_receipt(store, final_device, context, worker_result, dependencies)
        with store.write(closing=True):
            if finalization.exists():
                searchctl.require(json.loads(finalization.read_text()) == final_device, 'device finalization conflict')
            else:
                projectctl.atomic(finalization, final_device)
        usage = searchctl.evidence.device_usage(store, context, device)
        if usage and not usage['settled']:
            store.apply({'event_id': request_id + '-device-settle', 'op': 'control-settle',
                         'id': usage['id'], 'used': usage['used'], 'evidence': final_device['path']})
    process_path = runctl.location(store.workspace, store.project_id, run_id, request_id) / 'process-receipt.json'
    receipt = {key: context[key] for key in ('project_id', 'run_id', 'candidate_id', 'request_id', 'artifacts', 'evaluation_plan')}
    receipt.update(schema='operator-harness/execution-receipt@1', producer='evaluate@1',
                   backend=data['execution']['backend'], status='exited', exit_codes=process['exit_codes'],
                   valid=(accuracy == 'passed'),
                   process=store.reference(str(process_path.relative_to(store.root))))
    if final_device:
        receipt.update(device_receipt=final_device, worker_result=worker_result)
    report = {key: context[key] for key in ('project_id', 'run_id', 'candidate_id', 'candidate', 'artifacts', 'evaluation_plan')}
    report.update(schema='operator-harness/validation@1', receipts=[receipt],
                  cases=produced['cases'], diagnostics=produced['diagnostics'])
    validation = emission.save(output / 'validation.json', report) if context['operation'] != 'collect-only' else None
    performance = 'not_applicable' if data['performance_kind'] == 'none' else 'not_run'
    if accuracy == 'passed' and data['performance_kind'] != 'none':
        passed = []
        for row, case in zip(report['cases'], data['cases']):
            perf = row['performance']
            values = perf['samples_ms'] if 'samples_ms' in perf else json.loads(store.local(perf['path']).read_text())['samples_ms']
            aggregate = statistics.mean(values) if data['statistic'] == 'mean' else statistics.median(values)
            passed.append(aggregate <= case['max_latency_ms'])
        performance = 'passed' if all(passed) else 'failed'
    analysis = output / 'ANALYSIS_DATA.md'
    if not analysis.exists():
        diag_status = ('ready' if produced.get('diagnostics_verified') else 'partial') if report['diagnostics'] else 'not_applicable'
        analysis.write_text(render('Analysis data', {'project_id': store.project_id, 'run_id': run_id,
            'candidate_id': candidate_id, 'status': diag_status},
            '\n'.join(ref['path'] for ref in report['diagnostics'])))
    fields = {'contract': 'operator-harness/node-result@4', 'project_id': store.project_id,
              'run_id': run_id, 'node': run['stage'], 'search_node_id': run['search_node_id'],
              'status': 'completed' if accuracy == 'passed' else 'blocked',
              'candidate_manifest': candidate_path,
              'accuracy_status': accuracy, 'performance_status': performance,
              'analysis_data': str(analysis.relative_to(store.root)),
              'regression_cause': 'unknown' if performance == 'failed' and store.value(run['task'], 'operation') == 'implement-and-validate' else 'none'}
    if run.get('batch_id'):
        fields.update(batch_id=run['batch_id'], analysis_run=run['analysis_run'])
    if run.get('queue_item_id'):
        fields['queue_item_id'] = run['queue_item_id']
        if run.get('analysis_run'):
            fields['analysis_run'] = run['analysis_run']
    if context['operation'] == 'collect-only':
        diagnostic = emission.diagnostics(dict(context, workspace_root=str(store.root)),
            output / 'diagnostic-evidence.json', produced['diagnostics'])
        if final_device:
            path = store.local(diagnostic['path'])
            report = json.loads(path.read_text())
            report.update(device_receipt=final_device, worker_result=worker_result, process=receipt['process'])
            emission.save(path, report)
        fields.update(accuracy_status='not_run', performance_status='not_applicable',
                      diagnostic_evidence=diagnostic['path'], regression_cause='none')
    else:
        fields['validation_evidence'] = str(validation.relative_to(store.root))
    result = output / 'RESULT.generated.md'
    if not result.exists():
        result.write_text(render('Execution result', fields, 'Generated from the frozen evaluator. Add implementation/porting review before acceptance.\n'))
    return {'status': produced['status'], 'result': str(result.relative_to(store.root)),
            'validation': str(validation.relative_to(store.root)) if validation else None, 'backend': data['execution']['backend']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--owner')
    sub = parser.add_subparsers(dest='command', required=True)
    child = sub.add_parser('worker')
    child.add_argument('context', type=Path)
    execute = sub.add_parser('run')
    for field in ('run', 'candidate', 'request'):
        execute.add_argument('--' + field, required=True)
    execute.add_argument('--pool')
    execute.add_argument('--amount', type=float)
    execute.add_argument('--device-receipt')
    create = sub.add_parser('candidate')
    for field in ('run', 'id', 'source', 'build', 'launch'):
        create.add_argument('--' + field, required=True)
    create.add_argument('--parent')
    args = parser.parse_args()
    try:
        store = searchctl.Store(args.workspace, args.project, args.owner)
        if args.command == 'worker':
            worker(store, json.loads(args.context.read_text()))
            return 0
        if args.command == 'candidate':
            result = candidate(store, args.run, args.id, args.source, args.build, args.launch, args.parent)
        else:
            result = evaluate(store, args.run, args.candidate, args.request,
                              {'pool': args.pool, 'amount': args.amount} if args.pool else None,
                              args.device_receipt)
        print(json.dumps(result, indent=2))
        return 0 if result.get('status', 'complete') == 'complete' else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'evaluate: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
