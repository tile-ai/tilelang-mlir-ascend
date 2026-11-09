#!/usr/bin/env python3
"""Execute a frozen Python evaluation adapter and emit consumable run evidence."""
import argparse
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


def worker(store, context):
    """Runs only in the bounded child process; comparator errors propagate."""
    data, _ = searchctl.evidence.plan(store, context['evaluation_plan'])
    adapter = load_module(store, data['execution']['adapter'], 'harness_adapter')
    root = store.local(context['output'])
    if context.get('operation') == 'collect-only':
        captures = []
        verifier = getattr(adapter, 'verify_diagnostics', None)
        for index, case in enumerate(data['cases']):
            ctx = dict(context, workspace_root=str(store.root), case_id=case['id'])
            state = adapter.prepare(ctx, case)
            capture = adapter.diagnostics(state, case, root / f'case-{index}', ctx)
            if callable(verifier):
                verifier(capture, case)
            store.verify_ref(capture['evidence'])
            captures.append(capture['evidence'])
        emission.save(root / 'worker-result.json', {
            'status': 'complete', 'cases': [], 'diagnostics': captures,
            'diagnostics_verified': callable(verifier),
        })
        return
    reference = load_module(store, data['inputs']['reference'], 'harness_reference')
    comparator = load_module(store, data['inputs']['comparator'], 'harness_comparator')
    backend = data['execution']['backend']
    root = store.local(context['output'])
    rows, diagnostics = [], []
    # Finish correctness for ALL cases before any timing.
    states = []
    passed = True
    for index, case in enumerate(data['cases']):
        output = root / f'case-{index}'
        output.mkdir(parents=True, exist_ok=True)
        ctx = dict(context, workspace_root=str(store.root), case_id=case['id'])
        state = adapter.prepare(ctx, case)
        expected = reference.run(state)
        actual = adapter.run(state)
        adapter.synchronize(state)
        raw = emission.save(output / 'outputs.json', {'actual': serializable(actual), 'expected': serializable(expected)})
        outputs = comparator.compare(actual, expected, case)
        searchctl.require(isinstance(outputs, dict) and set(outputs) == set(case['outputs']) and
                          all(type(v) is bool for v in outputs.values()), 'comparator must return all named outputs')
        passed = passed and all(outputs.values())
        common = {key: ctx[key] for key in ('project_id', 'run_id', 'candidate_id', 'case_id', 'request_id', 'artifacts', 'evaluation_plan')}
        report = dict(common, schema='operator-harness/accuracy@1',
                      raw_files=[emission.reference(store.root, raw)], outputs=outputs)
        rows.append({'id': case['id'], 'accuracy': report})
        states.append((state, ctx, output, case))
    if not passed:
        emission.save(root / 'worker-result.json', {'status': 'accuracy_failed', 'cases': rows, 'diagnostics': []})
        return
    verifier = getattr(adapter, 'verify_diagnostics', None)
    for row, (state, ctx, output, case) in zip(rows, states):
        if data['performance_kind'] != 'none':
            if backend == 'torch_npu':
                from torch_npu_measure import measure
                options = adapter.measurement(state, case, output / 'measure')
                report = measure(**options, evidence_context=ctx)
                row['performance'] = report['evidence']
            else:
                samples = []
                for _ in range(case['min_samples']):
                    adapter.synchronize(state)
                    began = time.perf_counter_ns()
                    result = adapter.run(state)
                    adapter.synchronize(state)
                    samples.append((time.perf_counter_ns() - began) / 1e6)
                    del result
                raw = emission.save(output / 'samples.json', {'metric': 'cpu_wall_ms', 'samples_ms': samples})
                common = {key: ctx[key] for key in ('project_id', 'run_id', 'candidate_id', 'case_id', 'request_id', 'artifacts', 'evaluation_plan')}
                row['performance'] = dict(common, schema='operator-harness/performance@1',
                                          raw_files=[emission.reference(store.root, raw)], samples_ms=samples)
        if context['diagnostics_required']:
            capture = adapter.diagnostics(state, case, output / 'diagnostics', ctx)
            if callable(verifier):
                verifier(capture, case)
            ref = capture['evidence']
            store.verify_ref(ref)
            report = json.loads(store.local(ref['path']).read_text())
            searchctl.require(report['schema'] == 'operator-harness/diagnostic-evidence@1' and
                              report['candidate'] == context['candidate'], 'diagnostic collector identity mismatch')
            diagnostics.append(ref)
    emission.save(root / 'worker-result.json', {
        'status': 'complete', 'cases': rows, 'diagnostics': diagnostics,
        'diagnostics_verified': bool(diagnostics and callable(verifier)),
    })


def candidate(store, run_id, candidate_id, source, build, launch, parent_id):
    tree = store.load()
    run = tree['runs'][run_id]
    directory = store.records / 'runs' / run_id / 'candidates' / searchctl.identifier(candidate_id)
    artifacts = {key: store.reference(path) for key, path in (('source', source), ('build', build), ('launch', launch))}
    record = {'contract': 'operator-harness/candidate@2', 'project_id': store.project_id,
              'search_node_id': run['search_node_id'], 'candidate_id': candidate_id,
              'producer_run': run_id, 'parent_candidate': parent_id or 'not_applicable'}
    for key, ref in artifacts.items():
        record[key] = ref['path']
        record[key + '_sha256'] = ref['sha256']
    path = directory / 'CANDIDATE.md'
    text = render('Candidate', record)
    with store.write():
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


def evaluate(store, run_id, candidate_path, request_id, reservation=None):
    tree = store.load()
    store.verify_run(tree, run_id)
    run = tree['runs'][run_id]
    data = json.loads(store.local(run['evaluation_plan']['path']).read_text())
    dependencies = dict(run.get('evaluation_inputs') or searchctl.evidence.plan(store, run['evaluation_plan'])[1])
    searchctl.require(data.get('execution'), 'plan needs a frozen execution adapter')
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
    with store.write():
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
    process = runctl.execute(store.workspace, store.project_id, store.owner, spec)
    if process['status'] != 'exited' or process['exit_codes']['process'] != 0:
        return {'status': process['status'], 'process': process}
    produced = json.loads((output / 'worker-result.json').read_text())
    accuracy = 'passed' if produced['status'] == 'complete' else 'failed'
    process_path = runctl.location(store.workspace, store.project_id, run_id, request_id) / 'process-receipt.json'
    receipt = {key: context[key] for key in ('project_id', 'run_id', 'candidate_id', 'request_id', 'artifacts', 'evaluation_plan')}
    receipt.update(schema='operator-harness/execution-receipt@1', producer='evaluate@1',
                   backend=data['execution']['backend'], status='exited', exit_codes=process['exit_codes'],
                   valid=(accuracy == 'passed'),
                   process=store.reference(str(process_path.relative_to(store.root))))
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
    if context['operation'] == 'collect-only':
        diagnostic = emission.diagnostics(dict(context, workspace_root=str(store.root)),
            output / 'diagnostic-evidence.json', produced['diagnostics'])
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
                              {'pool': args.pool, 'amount': args.amount} if args.pool else None)
        print(json.dumps(result, indent=2))
        return 0 if result.get('status', 'complete') == 'complete' else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'evaluate: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
