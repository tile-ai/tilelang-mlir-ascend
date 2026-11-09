"""Emit workspace-relative evidence from actual collectors."""
import hashlib
import json
from pathlib import Path


def reference(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('artifact must be a file inside the workspace')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.relative_to(root)), 'sha256': digest.hexdigest()}


def files(root, directory):
    return [reference(root, p) for p in sorted(Path(directory).rglob('*')) if p.is_file()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return path


def case_report(context, kind, path, raw_files, **values):
    keys = ('project_id', 'run_id', 'candidate_id', 'case_id', 'request_id', 'artifacts', 'evaluation_plan')
    data = {key: context[key] for key in keys}
    data.update(schema='operator-harness/' + kind + '@1', raw_files=raw_files, **values)
    save(path, data)
    return reference(context['workspace_root'], path)


def diagnostics(context, path, raw_files):
    if not raw_files:
        raise ValueError('empty diagnostic capture')
    save(path, {'schema': 'operator-harness/diagnostic-evidence@1',
                'project_id': context['project_id'], 'run_id': context['run_id'],
                'candidate': context['candidate'], 'request_id': context['request_id'],
                'artifacts': raw_files})
    return reference(context['workspace_root'], path)
