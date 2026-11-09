"""Managed-file upgrade plans and recoverable rollback. No user-file deletion."""
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

START, END = '<!-- operator-harness -->', '<!-- /operator-harness -->'


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def metadata(root):
    path = root / '.agents/.harness'
    for component in (root / '.agents', path):
        if component.is_symlink():
            raise ValueError('installation metadata cannot traverse symlinks')
    return path


@contextmanager
def lock(root):
    directory = metadata(root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def target(root, relative):
    if relative != 'AGENTS.md' and not relative.startswith('.agents/skills/harness-'):
        raise ValueError('unmanaged upgrade target')
    path = root / relative
    if not path.resolve().is_relative_to(root) or '..' in Path(relative).parts:
        raise ValueError('upgrade target escapes workspace')
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError('source-linked development workspace: update assets directly; managed upgrades require copied files')
    return path


def block(data):
    text = data.decode()
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError('missing/ambiguous Harness AGENTS block')
    return text[text.index(START):text.index(END) + len(END)].encode()


def inventory(source, names):
    files = {}
    for name in names:
        for path in (source / name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files['.agents/skills/' + str(path.relative_to(source))] = path.read_bytes()
    return files


def manifest(files, entry):
    return {'schema': 'operator-harness/installation@1',
            'files': {key: digest(value) for key, value in files.items()}, 'entry_sha256': digest(entry)}


def record_install(root, source, names, entry):
    path = metadata(root) / 'manifest.json'
    data = manifest(inventory(source, names), entry)
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError('installed manifest differs; use an upgrade plan')
    atomic(path, encode(data))


def prepare(root, source, names, entry):
    path = metadata(root) / 'manifest.json'
    if not path.exists():
        raise ValueError('no managed installation manifest; verify/adopt the matching old bundle first')
    before_manifest = json.loads(path.read_text())
    if before_manifest.get('schema') != 'operator-harness/installation@1':
        raise ValueError('unsupported installation manifest')
    files = inventory(source, names)
    after_manifest = manifest(files, entry)
    operations = []
    for relative in sorted(set(files) | set(before_manifest['files'])):
        path = target(root, relative)
        before = path.read_bytes() if path.exists() else None
        expected = before_manifest['files'].get(relative)
        if digest(before) != expected:
            raise ValueError('user-modified or conflicting installed file: ' + relative)
        after = files.get(relative)
        if before != after:
            operations.append({'path': relative, 'before': before, 'after': after})
    path = target(root, 'AGENTS.md')
    before = path.read_bytes()
    old_block = block(before)
    if digest(old_block) != before_manifest['entry_sha256']:
        raise ValueError('user-modified Harness AGENTS block')
    after = before.replace(old_block, entry, 1)
    if before != after:
        operations.append({'path': 'AGENTS.md', 'before': before, 'after': after})
    public = {'before_manifest': digest(encode(before_manifest)), 'after_manifest': digest(encode(after_manifest)),
              'history': {p.name: digest(p.read_bytes()) for p in sorted((metadata(root) / 'transactions').glob('*.json'))},
              'changes': [{'path': op['path'], 'before': digest(op['before']), 'after': digest(op['after'])}
                          for op in operations]}
    public['id'] = digest(encode(public))
    return public, operations, before_manifest, after_manifest


def transaction(root, transaction_id, rollback=False):
    if len(transaction_id) != 64 or any(c not in '0123456789abcdef' for c in transaction_id):
        raise ValueError('invalid transaction ID')
    path = metadata(root) / 'transactions' / (transaction_id + '.json')
    journal = json.loads(path.read_text())
    rollback = rollback or journal.get('status') == 'rolling_back'
    before_key, after_key = ('after', 'before') if rollback else ('before', 'after')
    before_manifest = journal[before_key + '_manifest']
    after_manifest = journal[after_key + '_manifest']
    current_manifest = json.loads((metadata(root) / 'manifest.json').read_text())
    if current_manifest not in (before_manifest, after_manifest):
        raise ValueError('another version was installed; refuse stale transaction')
    if journal.get('status') == 'rolled_back' and not rollback:
        raise ValueError('transaction was rolled back; generate a new upgrade plan')
    decoded = []
    for op in journal['operations']:
        file = target(root, op['path'])
        old = base64.b64decode(op[before_key]) if op[before_key] is not None else None
        new = base64.b64decode(op[after_key]) if op[after_key] is not None else None
        actual = file.read_bytes() if file.exists() else None
        if actual not in (old, new):
            raise ValueError('file changed after upgrade plan: ' + op['path'])
        decoded.append((file, new))
    # Validate every target before writing any, then resume idempotently after interruption.
    journal["status"] = "rolling_back" if rollback else "applying"
    atomic(path, encode(journal))
    for file, new in decoded:
        if new is None:
            file.unlink(missing_ok=True)
        else:
            atomic(file, new)
    atomic(metadata(root) / 'manifest.json', encode(after_manifest))
    journal['status'] = 'rolled_back' if rollback else 'completed'
    atomic(path, encode(journal))
    return {'id': transaction_id, 'status': journal['status']}


def upgrade(root, source, names, entry, plan_id):
    public, operations, before, after = prepare(root, source, names, entry)
    if public['id'] != plan_id:
        raise ValueError('upgrade plan changed; review the new plan')
    path = metadata(root) / 'transactions' / (plan_id + '.json')
    journal = {'schema': 'operator-harness/install-transaction@1', 'status': 'prepared',
               'before_manifest': before, 'after_manifest': after,
               'operations': [{key: base64.b64encode(value).decode() if isinstance(value, bytes) else value
                               for key, value in op.items()} for op in operations]}
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get('status') == 'rolled_back':
            raise ValueError('transaction already rolled back; prepare a new plan')
        return transaction(root, plan_id)
    else:
        atomic(path, encode(journal))
    return transaction(root, plan_id)


def adopt(root, bundle_root):
    """Record only an exactly matching legacy bundle; never infer ownership by name alone."""
    source = Path(bundle_root).resolve(strict=True) / 'assets/skills'
    names = [p.name for p in source.glob('harness-*') if (p / 'SKILL.md').is_file()]
    if not names:
        raise ValueError('no skills in the supplied legacy bundle')
    entry = (source.parent / 'codex/AGENTS.md').read_text().strip().encode()
    files = inventory(source, names)
    for relative, content in files.items():
        file = target(root, relative)
        if not file.is_file() or file.read_bytes() != content:
            raise ValueError('legacy bundle does not match installed file: ' + relative)
    if block(target(root, 'AGENTS.md').read_bytes()) != entry:
        raise ValueError('legacy bundle does not match installed AGENTS block')
    record_install(root, source, names, entry)
    return {'status': 'adopted', 'files': len(files)}
