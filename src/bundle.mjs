import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  HOSTS,
  discoverInstallations,
  getHostLayout,
  getLayoutRoots,
  isManagedSkillName,
  layoutFromManifest,
  normalizeEol,
  normalizeScope,
  renderEntryForHost,
  renderSkillFileForHost,
} from './hosts.mjs';

export const START_MARKER = '<!-- operator-harness -->';
export const END_MARKER = '<!-- /operator-harness -->';
export const MANIFEST_SCHEMA_V1 = 'operator-harness/installation@1';
export const MANIFEST_SCHEMA_V2 = 'operator-harness/installation@2';
export const TRANSACTION_SCHEMA_V1 = 'operator-harness/install-transaction@1';

export const SKILL_NAMES = Object.freeze([
  'ascend950-simulation',
  'harness-new-project',
  'harness-onboard',
  'harness-resume',
  'harness-prepare-environment',
  'harness-establish-baseline',
  'harness-port-baseline',
  'harness-analysis',
  'harness-develop',
  'harness-status',
  'harness-next',
  'harness-automatic',
  'harness-archive-project',
  'harness-restore-project',
  'tilelang-ascend-generation',
  'tilelang-ascend-knowledge',
  'ascend-roofline',
]);

const PACKAGE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_ASSETS_ROOT = path.join(PACKAGE_ROOT, 'assets');
const SLEEP_BUFFER = new Int32Array(new SharedArrayBuffer(4));

export function getPackageInfo() {
  const pkgPath = path.join(PACKAGE_ROOT, 'package.json');
  const raw = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  return { name: raw.name, version: raw.version };
}

export function getDefaultSourceRoot() {
  return DEFAULT_ASSETS_ROOT;
}

export function sha256(data) {
  if (data === null || data === undefined) {
    return null;
  }
  const buf = Buffer.isBuffer(data) ? data : Buffer.from(data, 'utf8');
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function sortKeysDeep(value) {
  if (Array.isArray(value)) {
    return value.map(sortKeysDeep);
  }
  if (value && typeof value === 'object') {
    const out = {};
    for (const key of Object.keys(value).sort()) {
      out[key] = sortKeysDeep(value[key]);
    }
    return out;
  }
  return value;
}

export function encodeCanonicalJson(value) {
  return Buffer.from(`${JSON.stringify(sortKeysDeep(value), null, 2)}\n`, 'utf8');
}

function sleepMs(ms) {
  Atomics.wait(SLEEP_BUFFER, 0, 0, ms);
}

function lstatIfExists(targetPath) {
  try {
    return fs.lstatSync(targetPath);
  } catch (err) {
    if (err && err.code === 'ENOENT') {
      return null;
    }
    throw err;
  }
}

function ensureLayout(hostOrLayout, scope) {
  if (hostOrLayout && hostOrLayout.skillsDir && hostOrLayout.metadataDir && hostOrLayout.scope) {
    if (scope && normalizeScope(scope) !== hostOrLayout.scope) {
      return getHostLayout(hostOrLayout.id, scope);
    }
    return hostOrLayout;
  }
  return getHostLayout(hostOrLayout || HOSTS.codex, scope || 'local');
}

export function resolveDirectory(dirPath, label = 'target directory') {
  if (!dirPath || typeof dirPath !== 'string') {
    throw new Error(`${label} is required`);
  }
  const expanded = dirPath.startsWith('~/') || dirPath === '~'
    ? path.join(os.homedir(), dirPath.slice(1))
    : dirPath;
  const resolved = path.resolve(expanded);
  if (!fs.existsSync(resolved)) {
    throw new Error(`${label} does not exist: ${resolved}`);
  }
  const real = fs.realpathSync(resolved);
  if (!fs.statSync(real).isDirectory()) {
    throw new Error(`${label} must be an existing directory: ${real}`);
  }
  return real;
}

export function resolveWorkspaceRoot(workspace) {
  return resolveDirectory(workspace, 'workspace directory');
}

export function resolveBaseRoot({ scope = 'local', workspace = '.', homeDir = os.homedir() }) {
  const resolvedScope = normalizeScope(scope);
  if (resolvedScope === 'global') {
    return resolveDirectory(homeDir, 'home directory');
  }
  return resolveDirectory(workspace, 'target directory');
}

export function atomicWrite(filePath, data) {
  const buf = Buffer.isBuffer(data) ? data : Buffer.from(data, 'utf8');
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
  const tempPath = path.join(
    dir,
    `.pending-${process.pid}-${crypto.randomBytes(6).toString('hex')}`,
  );
  let fd = null;
  try {
    fd = fs.openSync(tempPath, 'wx');
    let offset = 0;
    while (offset < buf.length) {
      offset += fs.writeSync(fd, buf, offset, buf.length - offset);
    }
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fd = null;
    fs.renameSync(tempPath, filePath);
    if (process.platform !== 'win32') {
      try {
        const dirFd = fs.openSync(dir, 'r');
        try {
          fs.fsyncSync(dirFd);
        } finally {
          fs.closeSync(dirFd);
        }
      } catch (err) {
        if (!err || !['EINVAL', 'EPERM', 'EISDIR', 'ENOTSUP', 'EBADF'].includes(err.code)) {
          throw err;
        }
      }
    }
  } finally {
    if (fd !== null) {
      try {
        fs.closeSync(fd);
      } catch {
        // ignore close failure on cleanup
      }
    }
    if (fs.existsSync(tempPath)) {
      try {
        fs.unlinkSync(tempPath);
      } catch {
        // ignore cleanup failure
      }
    }
  }
}

function assertNoSymlinkPath(baseRoot, relativePosix, errorMessage) {
  const parts = relativePosix.split('/').filter(Boolean);
  let current = baseRoot;
  for (const part of parts) {
    current = path.join(current, part);
    const stat = lstatIfExists(current);
    if (stat && stat.isSymbolicLink()) {
      throw new Error(errorMessage);
    }
  }
}

export function getMetadataDir(baseRoot, layout) {
  assertNoSymlinkPath(
    baseRoot,
    layout.metadataDir,
    'installation metadata cannot traverse symlinks',
  );
  return path.join(baseRoot, ...layout.metadataDir.split('/'));
}

export function withLock(baseRoot, layout, fn) {
  const metaDir = getMetadataDir(baseRoot, layout);
  fs.mkdirSync(metaDir, { recursive: true });
  const lockPath = path.join(metaDir, 'lock');
  const lockStat = lstatIfExists(lockPath);
  if (lockStat && lockStat.isSymbolicLink()) {
    throw new Error('installation lock cannot be a symlink');
  }
  const deadline = Date.now() + 3000;
  let fd = null;
  while (true) {
    try {
      fd = fs.openSync(lockPath, 'wx');
      const payload = JSON.stringify({ pid: process.pid, created_at: new Date().toISOString() });
      fs.writeSync(fd, `${payload}\n`, 0, 'utf8');
      fs.closeSync(fd);
      fd = null;
      break;
    } catch (err) {
      if (fd !== null) {
        try {
          fs.closeSync(fd);
        } catch {
          // ignore
        }
        fd = null;
      }
      if (err && err.code === 'EEXIST' && Date.now() < deadline) {
        sleepMs(50);
        continue;
      }
      if (err && err.code === 'EEXIST') {
        throw new Error(`installation lock is busy (${lockPath}); wait for active operation or remove stale lock`);
      }
      throw err;
    }
  }
  try {
    return fn();
  } finally {
    try {
      fs.unlinkSync(lockPath);
    } catch (err) {
      if (!err || err.code !== 'ENOENT') {
        throw err;
      }
    }
  }
}

export function resolveManagedTarget(baseRoot, relativePosix, layout) {
  if (typeof relativePosix !== 'string' || !relativePosix || relativePosix.includes('\\')) {
    throw new Error(`invalid managed target path: ${relativePosix}`);
  }
  const parts = relativePosix.split('/');
  if (relativePosix.startsWith('/') || parts.some((p) => p === '..' || p === '.' || p === '')) {
    throw new Error(`upgrade target escapes installation root: ${relativePosix}`);
  }
  const skillPrefix = `${layout.skillsDir}/`;
  const skillSegments = layout.skillsDir.split('/');
  const isManagedSkillFile = relativePosix.startsWith(skillPrefix)
    && parts.length >= skillSegments.length + 2
    && isManagedSkillName(parts[skillSegments.length]);
  if (relativePosix !== layout.instructionFile && !isManagedSkillFile) {
    throw new Error(`unmanaged upgrade target: ${relativePosix}`);
  }
  const resolved = path.resolve(baseRoot, ...parts);
  const rel = path.relative(baseRoot, resolved);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) {
    throw new Error(`upgrade target escapes installation root: ${relativePosix}`);
  }
  assertNoSymlinkPath(
    baseRoot,
    relativePosix,
    'source-linked development workspace: update assets directly; managed upgrades require copied files',
  );
  return resolved;
}

function countOccurrences(haystack, needle) {
  let count = 0;
  let pos = 0;
  while (true) {
    const idx = haystack.indexOf(needle, pos);
    if (idx === -1) break;
    count += 1;
    pos = idx + needle.length;
  }
  return count;
}

export function extractHarnessBlock(text) {
  const startCount = countOccurrences(text, START_MARKER);
  const endCount = countOccurrences(text, END_MARKER);
  if (startCount === 0 && endCount === 0) {
    return null;
  }
  if (startCount !== 1 || endCount !== 1) {
    throw new Error('missing/ambiguous Harness AGENTS block; files preserved');
  }
  const startIdx = text.indexOf(START_MARKER);
  const endIdx = text.indexOf(END_MARKER);
  if (startIdx > endIdx) {
    throw new Error('missing/ambiguous Harness AGENTS block; files preserved');
  }
  const raw = text.slice(startIdx, endIdx + END_MARKER.length);
  return {
    raw,
    normalized: normalizeEol(raw),
    startIdx,
    endIdx: endIdx + END_MARKER.length,
  };
}

function matchFileEol(blockText, referenceText) {
  const usesCrlf = referenceText.includes('\r\n');
  return usesCrlf ? normalizeEol(blockText).replace(/\n/g, '\r\n') : normalizeEol(blockText);
}

export function loadSourceEntry(assetsRoot, layout) {
  const entryPath = path.join(assetsRoot, 'codex', 'AGENTS.md');
  const raw = fs.readFileSync(entryPath, 'utf8');
  return renderEntryForHost(raw, layout);
}

function walkFiles(dirPath, baseDir, out) {
  for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
    if (entry.name === '__pycache__' || entry.name.endsWith('.pyc') || entry.name.endsWith('.pyo')) {
      continue;
    }
    const full = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      walkFiles(full, baseDir, out);
    } else if (entry.isFile()) {
      const rel = path.relative(baseDir, full).split(path.sep).join('/');
      out.push({ rel, full });
    }
  }
}

export function collectInventory(skillsSourceDir, skillNames, layout) {
  const files = {};
  for (const name of skillNames) {
    const skillDir = path.join(skillsSourceDir, name);
    if (!fs.existsSync(skillDir)) {
      throw new Error(`missing bundled skill directory: ${skillDir}`);
    }
    const entries = [];
    walkFiles(skillDir, skillsSourceDir, entries);
    entries.sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));
    for (const { rel, full } of entries) {
      const rawBuf = fs.readFileSync(full);
      const rendered = renderSkillFileForHost(rel, rawBuf, layout);
      files[`${layout.skillsDir}/${rel}`] = rendered;
    }
  }
  return files;
}

export function buildManifest(filesMap, entryText, layout, pkgInfo = getPackageInfo()) {
  const digests = {};
  for (const key of Object.keys(filesMap).sort()) {
    digests[key] = sha256(filesMap[key]);
  }
  return {
    schema: MANIFEST_SCHEMA_V2,
    package_name: pkgInfo.name,
    package_version: pkgInfo.version,
    scope: layout.scope,
    host: layout.id,
    skills_dir: layout.skillsDir,
    instruction_file: layout.instructionFile,
    files: digests,
    entry_sha256: sha256(Buffer.from(normalizeEol(entryText).trim(), 'utf8')),
  };
}

export function parseManifest(rawManifest, fallbackLayout = getHostLayout(HOSTS.codex, 'local')) {
  if (!rawManifest || typeof rawManifest !== 'object') {
    throw new Error('invalid installation manifest');
  }
  const schema = rawManifest.schema;
  if (schema !== MANIFEST_SCHEMA_V1 && schema !== MANIFEST_SCHEMA_V2) {
    throw new Error(`unsupported installation manifest schema: ${schema}`);
  }
  if (!rawManifest.files || typeof rawManifest.files !== 'object' || Array.isArray(rawManifest.files)) {
    throw new Error('corrupted installation manifest: missing files map');
  }
  if (typeof rawManifest.entry_sha256 !== 'string' || !rawManifest.entry_sha256) {
    throw new Error('corrupted installation manifest: missing entry_sha256');
  }
  const layout = layoutFromManifest(rawManifest, fallbackLayout);
  return {
    raw: rawManifest,
    schema,
    scope: layout.scope,
    layout,
    skills_dir: rawManifest.skills_dir || layout.skillsDir,
    instruction_file: rawManifest.instruction_file || layout.instructionFile,
    package_name: rawManifest.package_name || null,
    package_version: rawManifest.package_version || null,
    files: rawManifest.files,
    entry_sha256: rawManifest.entry_sha256,
  };
}

function sameManagedContent(manifestA, manifestB) {
  if (manifestA.entry_sha256 !== manifestB.entry_sha256) {
    return false;
  }
  const keysA = Object.keys(manifestA.files).sort();
  const keysB = Object.keys(manifestB.files).sort();
  if (keysA.length !== keysB.length) {
    return false;
  }
  for (let i = 0; i < keysA.length; i += 1) {
    if (keysA[i] !== keysB[i] || manifestA.files[keysA[i]] !== manifestB.files[keysB[i]]) {
      return false;
    }
  }
  return true;
}

function pruneEmptyDirs(startDir, stopDir) {
  let current = startDir;
  const resolvedStop = path.resolve(stopDir);
  while (path.resolve(current) !== resolvedStop) {
    if (!fs.existsSync(current)) {
      current = path.dirname(current);
      continue;
    }
    const entries = fs.readdirSync(current);
    if (entries.length > 0) {
      break;
    }
    fs.rmdirSync(current);
    current = path.dirname(current);
  }
}

export function installWorkspace({
  workspace = '.',
  scope = 'local',
  homeDir = os.homedir(),
  host = HOSTS.codex,
  assetsRoot = DEFAULT_ASSETS_ROOT,
  skillNames = SKILL_NAMES,
  checkOnly = false,
  packageInfo = getPackageInfo(),
  atomicWriter = atomicWrite,
}) {
  const layout = ensureLayout(host, scope);
  const root = resolveBaseRoot({ scope: layout.scope, workspace, homeDir });
  const skillsSourceDir = path.join(assetsRoot, 'skills');
  const entryText = loadSourceEntry(assetsRoot, layout);
  const filesMap = collectInventory(skillsSourceDir, skillNames, layout);
  const expectedManifest = buildManifest(filesMap, entryText, layout, packageInfo);

  // Validate symlinks and conflicts BEFORE writing anything.
  assertNoSymlinkPath(
    root,
    layout.skillsDir,
    'source-linked development workspace: update assets/skills directly or install into a separate workspace',
  );
  const pendingSkills = [];
  for (const name of skillNames) {
    const relSkillDir = `${layout.skillsDir}/${name}`;
    const destDir = path.join(root, ...relSkillDir.split('/'));
    const stat = lstatIfExists(destDir);
    if (stat && stat.isSymbolicLink()) {
      throw new Error(
        'source-linked development workspace: update assets/skills directly or install into a separate workspace',
      );
    }
    const skillFiles = Object.entries(filesMap).filter(([rel]) => rel.startsWith(`${relSkillDir}/`));
    if (stat) {
      if (!stat.isDirectory()) {
        throw new Error(`existing skill differs; files preserved: ${destDir}`);
      }
      for (const [rel, expectedBuf] of skillFiles) {
        const targetFile = resolveManagedTarget(root, rel, layout);
        if (!fs.existsSync(targetFile) || !fs.statSync(targetFile).isFile()) {
          throw new Error(`existing skill differs; files preserved: ${destDir}`);
        }
        const actual = fs.readFileSync(targetFile);
        if (!actual.equals(expectedBuf)) {
          throw new Error(`existing skill differs; files preserved: ${destDir}`);
        }
      }
    } else {
      pendingSkills.push(name);
    }
  }

  const instructionPath = resolveManagedTarget(root, layout.instructionFile, layout);
  const existingInstruction = fs.existsSync(instructionPath)
    ? fs.readFileSync(instructionPath, 'utf8')
    : '';
  const existingBlock = extractHarnessBlock(existingInstruction);
  if (existingBlock && existingBlock.normalized !== entryText) {
    throw new Error(
      `existing Harness AGENTS block differs in ${layout.instructionFile}; review its upgrade before installing; files preserved`,
    );
  }

  const manifestPath = path.join(getMetadataDir(root, layout), 'manifest.json');
  if (fs.existsSync(manifestPath)) {
    const existingManifest = parseManifest(JSON.parse(fs.readFileSync(manifestPath, 'utf8')), layout);
    if (!sameManagedContent(existingManifest, expectedManifest)) {
      throw new Error('installed manifest differs; review an upgrade plan before installing');
    }
  }

  if (checkOnly) {
    return {
      status: pendingSkills.length > 0 ? 'ready' : 'already_installed',
      scope: layout.scope,
      host: layout.id,
      install_root: root,
      skills_dir: path.join(root, ...layout.skillsDir.split('/')),
      instruction_file: instructionPath,
      installed_skills: pendingSkills,
    };
  }

  return withLock(root, layout, () => {
    for (const name of pendingSkills) {
      const prefix = `${layout.skillsDir}/${name}/`;
      for (const [rel, buf] of Object.entries(filesMap)) {
        if (!rel.startsWith(prefix)) continue;
        const targetFile = resolveManagedTarget(root, rel, layout);
        atomicWriter(targetFile, buf);
      }
    }

    if (!existingBlock) {
      const eol = existingInstruction.includes('\r\n') ? '\r\n' : '\n';
      const rendered = matchFileEol(entryText, existingInstruction);
      const separator = existingInstruction
        ? (existingInstruction.endsWith('\n') ? eol : `${eol}${eol}`)
        : '';
      atomicWriter(
        instructionPath,
        Buffer.from(`${existingInstruction}${separator}${rendered}${eol}`, 'utf8'),
      );
    }

    atomicWriter(manifestPath, encodeCanonicalJson(expectedManifest));

    return {
      status: pendingSkills.length > 0 ? 'installed' : 'already_installed',
      scope: layout.scope,
      host: layout.id,
      install_root: root,
      skills_dir: path.join(root, ...layout.skillsDir.split('/')),
      instruction_file: instructionPath,
      installed_skills: pendingSkills,
      manifest_path: manifestPath,
    };
  });
}

export function prepareUpgrade({
  workspace = '.',
  scope = 'local',
  homeDir = os.homedir(),
  host = HOSTS.codex,
  assetsRoot = DEFAULT_ASSETS_ROOT,
  skillNames = SKILL_NAMES,
  packageInfo = getPackageInfo(),
}) {
  const layout = ensureLayout(host, scope);
  const root = resolveBaseRoot({ scope: layout.scope, workspace, homeDir });
  const metaDir = getMetadataDir(root, layout);
  const manifestPath = path.join(metaDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    throw new Error('no managed installation manifest; verify/adopt the matching old bundle first');
  }
  const beforeParsed = parseManifest(JSON.parse(fs.readFileSync(manifestPath, 'utf8')), layout);
  const beforeManifest = beforeParsed.raw;
  const effectiveLayout = beforeParsed.layout;

  const skillsSourceDir = path.join(assetsRoot, 'skills');
  const entryText = loadSourceEntry(assetsRoot, effectiveLayout);
  const filesMap = collectInventory(skillsSourceDir, skillNames, effectiveLayout);
  const afterManifest = buildManifest(filesMap, entryText, effectiveLayout, packageInfo);

  const operations = [];
  const allPaths = Array.from(
    new Set([...Object.keys(filesMap), ...Object.keys(beforeManifest.files)]),
  ).sort();

  for (const relative of allPaths) {
    const targetFile = resolveManagedTarget(root, relative, effectiveLayout);
    const before = fs.existsSync(targetFile) ? fs.readFileSync(targetFile) : null;
    const expected = Object.prototype.hasOwnProperty.call(beforeManifest.files, relative)
      ? beforeManifest.files[relative]
      : null;
    if (sha256(before) !== expected) {
      throw new Error(`user-modified or conflicting installed file: ${relative}`);
    }
    const after = Object.prototype.hasOwnProperty.call(filesMap, relative)
      ? filesMap[relative]
      : null;
    const changed = (before === null) !== (after === null) || (before !== null && !before.equals(after));
    if (changed) {
      operations.push({ path: relative, before, after });
    }
  }

  const instructionFile = resolveManagedTarget(root, effectiveLayout.instructionFile, effectiveLayout);
  if (!fs.existsSync(instructionFile)) {
    throw new Error(`missing instruction file: ${effectiveLayout.instructionFile}`);
  }
  const beforeInstructionBuf = fs.readFileSync(instructionFile);
  const beforeInstructionText = beforeInstructionBuf.toString('utf8');
  const oldBlock = extractHarnessBlock(beforeInstructionText);
  if (!oldBlock) {
    throw new Error(`missing/ambiguous Harness AGENTS block in ${effectiveLayout.instructionFile}`);
  }
  if (sha256(Buffer.from(oldBlock.normalized, 'utf8')) !== beforeManifest.entry_sha256) {
    throw new Error(`user-modified Harness AGENTS block in ${effectiveLayout.instructionFile}`);
  }
  const replacementBlock = matchFileEol(entryText, beforeInstructionText);
  const afterInstructionText =
    beforeInstructionText.slice(0, oldBlock.startIdx) +
    replacementBlock +
    beforeInstructionText.slice(oldBlock.endIdx);
  const afterInstructionBuf = Buffer.from(afterInstructionText, 'utf8');
  if (!beforeInstructionBuf.equals(afterInstructionBuf)) {
    operations.push({
      path: effectiveLayout.instructionFile,
      before: beforeInstructionBuf,
      after: afterInstructionBuf,
    });
  }

  const txDir = path.join(metaDir, 'transactions');
  const history = {};
  if (fs.existsSync(txDir)) {
    for (const fileName of fs.readdirSync(txDir).filter((f) => f.endsWith('.json')).sort()) {
      history[fileName] = sha256(fs.readFileSync(path.join(txDir, fileName)));
    }
  }

  const publicPlan = {
    scope: effectiveLayout.scope,
    host: effectiveLayout.id,
    before_manifest: sha256(encodeCanonicalJson(beforeManifest)),
    after_manifest: sha256(encodeCanonicalJson(afterManifest)),
    history,
    changes: operations.map((op) => ({
      path: op.path,
      before: sha256(op.before),
      after: sha256(op.after),
    })),
  };
  publicPlan.id = sha256(encodeCanonicalJson(publicPlan));

  return {
    plan: publicPlan,
    operations,
    beforeManifest,
    afterManifest,
    layout: effectiveLayout,
  };
}

function executeTransactionUnlocked({
  root,
  layout,
  transactionId,
  rollback = false,
  atomicWriter = atomicWrite,
}) {
  if (typeof transactionId !== 'string' || !/^[0-9a-f]{64}$/.test(transactionId)) {
    throw new Error('invalid transaction ID');
  }
  const metaDir = getMetadataDir(root, layout);
  const txPath = path.join(metaDir, 'transactions', `${transactionId}.json`);
  if (!fs.existsSync(txPath)) {
    throw new Error(`transaction not found: ${transactionId}`);
  }
  const journal = JSON.parse(fs.readFileSync(txPath, 'utf8'));
  const isRollback = Boolean(rollback) || journal.status === 'rolling_back';
  const beforeKey = isRollback ? 'after' : 'before';
  const afterKey = isRollback ? 'before' : 'after';
  const beforeManifest = journal[`${beforeKey}_manifest`];
  const afterManifest = journal[`${afterKey}_manifest`];

  const manifestPath = path.join(metaDir, 'manifest.json');
  const currentManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const currentDigest = sha256(encodeCanonicalJson(currentManifest));
  const beforeDigest = sha256(encodeCanonicalJson(beforeManifest));
  const afterDigest = sha256(encodeCanonicalJson(afterManifest));
  if (currentDigest !== beforeDigest && currentDigest !== afterDigest) {
    throw new Error('another version was installed; refuse stale transaction');
  }
  if (journal.status === 'rolled_back' && !isRollback) {
    throw new Error('transaction was rolled back; generate a new upgrade plan');
  }

  const effectiveLayout = parseManifest(afterManifest, layout).layout;
  const decoded = [];
  for (const op of journal.operations) {
    const targetFile = resolveManagedTarget(root, op.path, effectiveLayout);
    const oldBuf = op[beforeKey] !== null ? Buffer.from(op[beforeKey], 'base64') : null;
    const newBuf = op[afterKey] !== null ? Buffer.from(op[afterKey], 'base64') : null;
    const actualBuf = fs.existsSync(targetFile) ? fs.readFileSync(targetFile) : null;
    const matchesOld = (actualBuf === null && oldBuf === null)
      || (actualBuf !== null && oldBuf !== null && actualBuf.equals(oldBuf));
    const matchesNew = (actualBuf === null && newBuf === null)
      || (actualBuf !== null && newBuf !== null && actualBuf.equals(newBuf));
    if (!matchesOld && !matchesNew) {
      throw new Error(`file changed after upgrade plan: ${op.path}`);
    }
    decoded.push({ file: targetFile, rel: op.path, data: newBuf });
  }

  journal.status = isRollback ? 'rolling_back' : 'applying';
  atomicWriter(txPath, encodeCanonicalJson(journal));

  const skillsRoot = path.join(root, ...effectiveLayout.skillsDir.split('/'));
  for (const { file, rel, data } of decoded) {
    if (data === null) {
      if (fs.existsSync(file)) {
        fs.unlinkSync(file);
      }
      if (rel.startsWith(`${effectiveLayout.skillsDir}/`)) {
        pruneEmptyDirs(path.dirname(file), skillsRoot);
      }
    } else {
      atomicWriter(file, data);
    }
  }

  atomicWriter(manifestPath, encodeCanonicalJson(afterManifest));
  journal.status = isRollback ? 'rolled_back' : 'completed';
  atomicWriter(txPath, encodeCanonicalJson(journal));

  return {
    id: transactionId,
    scope: effectiveLayout.scope,
    host: effectiveLayout.id,
    status: journal.status,
  };
}

export function applyUpgrade({
  workspace = '.',
  scope = 'local',
  homeDir = os.homedir(),
  host = HOSTS.codex,
  planId,
  assetsRoot = DEFAULT_ASSETS_ROOT,
  skillNames = SKILL_NAMES,
  packageInfo = getPackageInfo(),
  atomicWriter = atomicWrite,
}) {
  const layout = ensureLayout(host, scope);
  const root = resolveBaseRoot({ scope: layout.scope, workspace, homeDir });
  return withLock(root, layout, () => {
    const prepared = prepareUpgrade({
      workspace: root,
      scope: layout.scope,
      homeDir: root,
      host: layout,
      assetsRoot,
      skillNames,
      packageInfo,
    });
    if (prepared.plan.id !== planId) {
      throw new Error('upgrade plan changed; review the new plan');
    }
    const metaDir = getMetadataDir(root, prepared.layout);
    const txPath = path.join(metaDir, 'transactions', `${planId}.json`);
    if (fs.existsSync(txPath)) {
      const prior = JSON.parse(fs.readFileSync(txPath, 'utf8'));
      if (prior.status === 'rolled_back') {
        throw new Error('transaction already rolled back; prepare a new plan');
      }
      return executeTransactionUnlocked({
        root,
        layout: prepared.layout,
        transactionId: planId,
        rollback: false,
        atomicWriter,
      });
    }
    const journal = {
      schema: TRANSACTION_SCHEMA_V1,
      status: 'prepared',
      before_manifest: prepared.beforeManifest,
      after_manifest: prepared.afterManifest,
      operations: prepared.operations.map((op) => ({
        path: op.path,
        before: op.before !== null ? op.before.toString('base64') : null,
        after: op.after !== null ? op.after.toString('base64') : null,
      })),
    };
    atomicWriter(txPath, encodeCanonicalJson(journal));
    return executeTransactionUnlocked({
      root,
      layout: prepared.layout,
      transactionId: planId,
      rollback: false,
      atomicWriter,
    });
  });
}

export function runTransaction({
  workspace = '.',
  scope = 'local',
  homeDir = os.homedir(),
  host = HOSTS.codex,
  transactionId,
  rollback = false,
  atomicWriter = atomicWrite,
}) {
  const layout = ensureLayout(host, scope);
  const root = resolveBaseRoot({ scope: layout.scope, workspace, homeDir });
  return withLock(root, layout, () => executeTransactionUnlocked({
    root,
    layout,
    transactionId,
    rollback,
    atomicWriter,
  }));
}

export function adoptLegacy({
  workspace = '.',
  scope = 'local',
  homeDir = os.homedir(),
  host = HOSTS.codex,
  bundleRoot,
  packageInfo = getPackageInfo(),
  atomicWriter = atomicWrite,
}) {
  const layout = ensureLayout(host, scope);
  const root = resolveBaseRoot({ scope: layout.scope, workspace, homeDir });
  const resolvedBundle = resolveDirectory(bundleRoot, 'legacy bundle');
  const skillsSourceDir = path.join(resolvedBundle, 'assets', 'skills');
  if (!fs.existsSync(skillsSourceDir) || !fs.statSync(skillsSourceDir).isDirectory()) {
    throw new Error('no skills in the supplied legacy bundle');
  }
  const names = fs.readdirSync(skillsSourceDir)
    .filter((name) => isManagedSkillName(name)
      && fs.existsSync(path.join(skillsSourceDir, name, 'SKILL.md')))
    .sort();
  if (names.length === 0) {
    throw new Error('no skills in the supplied legacy bundle');
  }
  const entryText = loadSourceEntry(path.join(resolvedBundle, 'assets'), layout);
  const filesMap = collectInventory(skillsSourceDir, names, layout);

  return withLock(root, layout, () => {
    for (const [relative, content] of Object.entries(filesMap)) {
      const targetFile = resolveManagedTarget(root, relative, layout);
      if (!fs.existsSync(targetFile) || !fs.statSync(targetFile).isFile()) {
        throw new Error(`legacy bundle does not match installed file: ${relative}`);
      }
      if (!fs.readFileSync(targetFile).equals(content)) {
        throw new Error(`legacy bundle does not match installed file: ${relative}`);
      }
    }
    const instructionFile = resolveManagedTarget(root, layout.instructionFile, layout);
    if (!fs.existsSync(instructionFile)) {
      throw new Error('legacy bundle does not match installed AGENTS block');
    }
    const installedBlock = extractHarnessBlock(fs.readFileSync(instructionFile, 'utf8'));
    if (!installedBlock || installedBlock.normalized !== entryText) {
      throw new Error('legacy bundle does not match installed AGENTS block');
    }
    const manifestPath = path.join(getMetadataDir(root, layout), 'manifest.json');
    const manifestData = buildManifest(filesMap, entryText, layout, packageInfo);
    atomicWriter(manifestPath, encodeCanonicalJson(manifestData));
    return {
      status: 'adopted',
      scope: layout.scope,
      host: layout.id,
      files: Object.keys(filesMap).length,
      manifest_path: manifestPath,
    };
  });
}

function stripHarnessBlock(existingText, blockInfo) {
  const before = existingText.slice(0, blockInfo.startIdx).replace(/[ \t]*(\r?\n){1,2}$/, '');
  const after = existingText.slice(blockInfo.endIdx).replace(/^(\r?\n){1,2}/, '');
  if (!before.trim() && !after.trim()) {
    return null;
  }
  const eol = existingText.includes('\r\n') ? '\r\n' : '\n';
  if (!before.trim()) {
    return `${after.replace(/^(\r?\n)+/, '')}`;
  }
  if (!after.trim()) {
    return `${before}${eol}`;
  }
  return `${before}${eol}${eol}${after}`;
}

export function uninstallWorkspace({
  workspace = '.',
  scope = 'local',
  homeDir = os.homedir(),
  host = HOSTS.codex,
  dryRun = false,
  force = false,
  atomicWriter = atomicWrite,
}) {
  const layout = ensureLayout(host, scope);
  const root = resolveBaseRoot({ scope: layout.scope, workspace, homeDir });
  const metaDir = getMetadataDir(root, layout);
  const manifestPath = path.join(metaDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`no managed ${layout.scope} installation found for host "${layout.id}" in ${root}`);
  }
  const parsed = parseManifest(JSON.parse(fs.readFileSync(manifestPath, 'utf8')), layout);
  const effectiveLayout = parsed.layout;

  const filesToRemove = [];
  for (const [relative, expectedDigest] of Object.entries(parsed.files).sort()) {
    const targetFile = resolveManagedTarget(root, relative, effectiveLayout);
    if (!fs.existsSync(targetFile)) {
      continue;
    }
    const actualDigest = sha256(fs.readFileSync(targetFile));
    if (actualDigest !== expectedDigest && !force) {
      throw new Error(
        `user-modified managed file refuses uninstall (pass --force to override): ${relative}`,
      );
    }
    filesToRemove.push({ relative, fullPath: targetFile });
  }

  const instructionFile = resolveManagedTarget(root, effectiveLayout.instructionFile, effectiveLayout);
  let instructionAction = 'unchanged';
  let nextInstructionContent = null;
  if (fs.existsSync(instructionFile)) {
    const existingText = fs.readFileSync(instructionFile, 'utf8');
    const blockInfo = extractHarnessBlock(existingText);
    if (blockInfo) {
      if (sha256(Buffer.from(blockInfo.normalized, 'utf8')) !== parsed.entry_sha256 && !force) {
        throw new Error(
          `user-modified Harness block in ${effectiveLayout.instructionFile} refuses uninstall (pass --force to override)`,
        );
      }
      nextInstructionContent = stripHarnessBlock(existingText, blockInfo);
      instructionAction = nextInstructionContent === null ? 'delete' : 'strip_block';
    }
  }

  if (dryRun) {
    return {
      status: 'dry_run',
      scope: effectiveLayout.scope,
      host: effectiveLayout.id,
      install_root: root,
      files_to_remove: filesToRemove.map((f) => f.relative),
      instruction_file: effectiveLayout.instructionFile,
      instruction_action: instructionAction,
      workspace_preserved: true,
    };
  }

  const skillsDirPath = path.join(root, ...effectiveLayout.skillsDir.split('/'));
  const pruneStopDir = effectiveLayout.scope === 'global'
    ? path.join(root, ...effectiveLayout.rootDir.split('/'))
    : root;

  const result = withLock(root, effectiveLayout, () => {
    for (const { fullPath } of filesToRemove) {
      if (fs.existsSync(fullPath)) {
        fs.unlinkSync(fullPath);
      }
      pruneEmptyDirs(path.dirname(fullPath), skillsDirPath);
    }

    if (instructionAction === 'delete' && fs.existsSync(instructionFile)) {
      fs.unlinkSync(instructionFile);
    } else if (instructionAction === 'strip_block') {
      atomicWriter(instructionFile, Buffer.from(nextInstructionContent, 'utf8'));
    }

    if (fs.existsSync(manifestPath)) {
      fs.unlinkSync(manifestPath);
    }
    const txDir = path.join(metaDir, 'transactions');
    if (fs.existsSync(txDir)) {
      fs.rmSync(txDir, { recursive: true, force: true });
    }

    return {
      status: 'uninstalled',
      scope: effectiveLayout.scope,
      host: effectiveLayout.id,
      install_root: root,
      removed_files: filesToRemove.length,
      instruction_file: effectiveLayout.instructionFile,
      instruction_action: instructionAction,
      workspace_preserved: true,
    };
  });

  // In global mode, never delete top-level ~/.codex / ~/.claude / ~/.gemini / ~/.agent dirs;
  // only prune empty .harness and empty skills subdirectories inside them.
  pruneEmptyDirs(metaDir, pruneStopDir);
  pruneEmptyDirs(skillsDirPath, pruneStopDir);
  return result;
}

function detectPython() {
  const candidates = process.platform === 'win32'
    ? [
        { cmd: 'py', args: ['-3', '--version'], display: 'py -3' },
        { cmd: 'python3', args: ['--version'], display: 'python3' },
        { cmd: 'python', args: ['--version'], display: 'python' },
      ]
    : [
        { cmd: 'python3', args: ['--version'], display: 'python3' },
        { cmd: 'python', args: ['--version'], display: 'python' },
      ];

  for (const candidate of candidates) {
    const res = spawnSync(candidate.cmd, candidate.args, {
      encoding: 'utf8',
      timeout: 5000,
    });
    if (res.error || res.status !== 0) continue;
    const output = `${res.stdout || ''} ${res.stderr || ''}`.trim();
    const match = output.match(/Python\s+(\d+)\.(\d+)\.(\d+)/i);
    if (!match) continue;
    const major = Number(match[1]);
    const minor = Number(match[2]);
    if (major >= 3) {
      return {
        available: true,
        command: candidate.display,
        version: `${match[1]}.${match[2]}.${match[3]}`,
        supported: major > 3 || minor >= 9,
      };
    }
  }
  return { available: false, command: null, version: null, supported: false };
}

function checkWorkspaceGitIgnore(dir) {
  const hasWorkspaceDir = fs.existsSync(path.join(dir, '.workspace'));
  const insideRes = spawnSync('git', ['rev-parse', '--is-inside-work-tree'], {
    cwd: dir,
    encoding: 'utf8',
    timeout: 5000,
  });
  if (!insideRes.error && insideRes.status === 0 && insideRes.stdout.trim() === 'true') {
    const ignoreRes = spawnSync('git', ['check-ignore', '-q', '--', '.workspace/'], {
      cwd: dir,
      encoding: 'utf8',
      timeout: 5000,
    });
    const ignored = ignoreRes.status === 0;
    let excludePath = null;
    const absRes = spawnSync('git', ['rev-parse', '--path-format=absolute', '--git-path', 'info/exclude'], {
      cwd: dir,
      encoding: 'utf8',
      timeout: 5000,
    });
    if (!absRes.error && absRes.status === 0 && absRes.stdout.trim()) {
      excludePath = absRes.stdout.trim();
    } else {
      const relRes = spawnSync('git', ['rev-parse', '--git-path', 'info/exclude'], {
        cwd: dir,
        encoding: 'utf8',
        timeout: 5000,
      });
      if (!relRes.error && relRes.status === 0 && relRes.stdout.trim()) {
        excludePath = path.resolve(dir, relRes.stdout.trim());
      }
    }
    return {
      is_git_repo: true,
      has_workspace_dir: hasWorkspaceDir,
      ignored,
      exclude_path: excludePath,
    };
  }

  // Fallback when git CLI is unavailable or directory is a non-git checkout.
  const gitDir = path.join(dir, '.git');
  const isGitRepo = fs.existsSync(gitDir);
  const containsWorkspaceRule = (filePath) => {
    if (!fs.existsSync(filePath)) return false;
    const lines = normalizeEol(fs.readFileSync(filePath, 'utf8')).split('\n').map((l) => l.trim());
    return lines.includes('.workspace/') || lines.includes('.workspace');
  };
  const inGitignore = containsWorkspaceRule(path.join(dir, '.gitignore'));
  let excludePath = null;
  let inInfoExclude = false;
  if (isGitRepo && fs.statSync(gitDir).isDirectory()) {
    excludePath = path.join(gitDir, 'info', 'exclude');
    inInfoExclude = containsWorkspaceRule(excludePath);
  }
  return {
    is_git_repo: isGitRepo,
    has_workspace_dir: hasWorkspaceDir,
    ignored: inGitignore || inInfoExclude,
    exclude_path: excludePath,
  };
}

function inspectLayout(root, layout, assetsRoot, skillNames, packageInfo) {
  const skillsDirPath = path.join(root, ...layout.skillsDir.split('/'));
  const hasSymlinkedSkill = skillNames.some((name) => {
    const st = lstatIfExists(path.join(skillsDirPath, name));
    return Boolean(st && st.isSymbolicLink());
  });
  if (hasSymlinkedSkill) {
    return {
      scope: layout.scope,
      host: layout.id,
      install_root: root,
      skills_dir: layout.skillsDir,
      instruction_file: layout.instructionFile,
      mode: 'source_linked_dev',
      ok: true,
      issues: [],
      warnings: ['source-linked development workspace detected; skills track assets/skills directly'],
    };
  }

  const metaDir = path.join(root, ...layout.metadataDir.split('/'));
  const manifestPath = path.join(metaDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    return {
      scope: layout.scope,
      host: layout.id,
      install_root: root,
      skills_dir: layout.skillsDir,
      instruction_file: layout.instructionFile,
      mode: 'not_installed',
      ok: false,
      issues: [`not installed (${layout.scope}) for host "${layout.id}" (missing ${layout.metadataDir}/manifest.json)`],
      warnings: [],
    };
  }

  const issues = [];
  const warnings = [];
  let parsed;
  try {
    parsed = parseManifest(JSON.parse(fs.readFileSync(manifestPath, 'utf8')), layout);
  } catch (err) {
    return {
      scope: layout.scope,
      host: layout.id,
      install_root: root,
      skills_dir: layout.skillsDir,
      instruction_file: layout.instructionFile,
      mode: 'corrupted',
      ok: false,
      issues: [`manifest error: ${err.message}`],
      warnings: [],
    };
  }

  const effectiveLayout = parsed.layout;
  let missingFiles = 0;
  let modifiedFiles = 0;
  for (const [relative, expectedDigest] of Object.entries(parsed.files)) {
    let targetFile;
    try {
      targetFile = resolveManagedTarget(root, relative, effectiveLayout);
    } catch (err) {
      issues.push(err.message);
      continue;
    }
    if (!fs.existsSync(targetFile)) {
      missingFiles += 1;
      issues.push(`missing managed file: ${relative}`);
    } else if (sha256(fs.readFileSync(targetFile)) !== expectedDigest) {
      modifiedFiles += 1;
      issues.push(`modified managed file: ${relative}`);
    }
  }

  const instructionPath = path.join(root, ...effectiveLayout.instructionFile.split('/'));
  if (!fs.existsSync(instructionPath)) {
    issues.push(`missing instruction file: ${effectiveLayout.instructionFile}`);
  } else {
    try {
      const blockInfo = extractHarnessBlock(fs.readFileSync(instructionPath, 'utf8'));
      if (!blockInfo) {
        issues.push(`missing Harness block in ${effectiveLayout.instructionFile}`);
      } else if (sha256(Buffer.from(blockInfo.normalized, 'utf8')) !== parsed.entry_sha256) {
        issues.push(`modified Harness block in ${effectiveLayout.instructionFile}`);
      }
    } catch (err) {
      issues.push(err.message);
    }
  }

  const txDir = path.join(metaDir, 'transactions');
  const pendingTransactions = [];
  if (fs.existsSync(txDir)) {
    for (const fileName of fs.readdirSync(txDir).filter((f) => f.endsWith('.json')).sort()) {
      try {
        const tx = JSON.parse(fs.readFileSync(path.join(txDir, fileName), 'utf8'));
        if (['prepared', 'applying', 'rolling_back'].includes(tx.status)) {
          pendingTransactions.push({ id: fileName.replace(/\.json$/, ''), status: tx.status });
          issues.push(
            `interrupted transaction ${fileName.replace(/\.json$/, '')} (status=${tx.status}); run "tilelang-harness resume"`,
          );
        }
      } catch (err) {
        issues.push(`corrupted transaction log ${fileName}: ${err.message}`);
      }
    }
  }

  let upgradeAvailable = false;
  try {
    const currentFiles = collectInventory(path.join(assetsRoot, 'skills'), skillNames, effectiveLayout);
    const currentEntry = loadSourceEntry(assetsRoot, effectiveLayout);
    const expectedManifest = buildManifest(currentFiles, currentEntry, effectiveLayout, packageInfo);
    const scopeFlag = effectiveLayout.scope === 'global' ? '--global' : '--local';
    if (!sameManagedContent(parsed, expectedManifest)) {
      upgradeAvailable = true;
      warnings.push(
        `installed bundle differs from tilelang-harness@${packageInfo.version}; run "tilelang-harness upgrade --plan ${scopeFlag} --host ${effectiveLayout.id}"`,
      );
    } else if (parsed.schema === MANIFEST_SCHEMA_V1) {
      warnings.push(
        `legacy installation@1 manifest detected; run "tilelang-harness upgrade --plan ${scopeFlag} --host ${effectiveLayout.id}" to refresh metadata`,
      );
    }
  } catch (err) {
    warnings.push(`could not compare with current bundle: ${err.message}`);
  }

  return {
    scope: effectiveLayout.scope,
    host: effectiveLayout.id,
    install_root: root,
    skills_dir: effectiveLayout.skillsDir,
    instruction_file: effectiveLayout.instructionFile,
    mode: 'managed',
    schema: parsed.schema,
    package_version: parsed.package_version,
    managed_files: Object.keys(parsed.files).length,
    missing_files: missingFiles,
    modified_files: modifiedFiles,
    pending_transactions: pendingTransactions,
    upgrade_available: upgradeAvailable,
    ok: issues.length === 0,
    issues,
    warnings,
  };
}

export function doctorWorkspace({
  workspace = '.',
  scope = 'local',
  homeDir = os.homedir(),
  host = null,
  assetsRoot = DEFAULT_ASSETS_ROOT,
  skillNames = SKILL_NAMES,
  packageInfo = getPackageInfo(),
}) {
  const nodeMajor = Number(process.versions.node.split('.')[0]);
  const python = detectPython();
  const warnings = [];
  const issues = [];

  if (nodeMajor < 18) {
    issues.push(`Node.js >= 18 required (found ${process.versions.node})`);
  }
  if (!python.available) {
    warnings.push(
      'Python 3 not found in PATH; install Python 3.9+ locally to run bundled skill control scripts (projectctl.py / searchctl.py)',
    );
  } else if (!python.supported) {
    warnings.push(`Python 3.9+ recommended for skill scripts (found ${python.version})`);
  }

  const scopesToCheck = scope === 'auto' ? ['local', 'global'] : [normalizeScope(scope)];
  const localRoot = scopesToCheck.includes('local') ? resolveDirectory(workspace, 'target directory') : null;
  const globalRoot = scopesToCheck.includes('global') ? resolveDirectory(homeDir, 'home directory') : null;

  const gitWorkspaceIgnore = localRoot ? checkWorkspaceGitIgnore(localRoot) : null;
  if (gitWorkspaceIgnore && gitWorkspaceIgnore.is_git_repo && gitWorkspaceIgnore.has_workspace_dir && !gitWorkspaceIgnore.ignored) {
    const excludeTarget = gitWorkspaceIgnore.exclude_path
      ? `"${gitWorkspaceIgnore.exclude_path}"`
      : '"$(git rev-parse --path-format=absolute --git-path info/exclude)"';
    warnings.push(
      `.workspace/ exists in this Git repo but is not ignored; run 'echo ".workspace/" >> ${excludeTarget}' (or add to .gitignore for team-wide use)`,
    );
  }

  const installations = [];
  for (const currentScope of scopesToCheck) {
    const baseRoot = currentScope === 'global' ? globalRoot : localRoot;
    if (host) {
      const layout = ensureLayout(host, currentScope);
      const inspected = inspectLayout(baseRoot, layout, assetsRoot, skillNames, packageInfo);
      if (scope !== 'auto' || inspected.mode !== 'not_installed') {
        installations.push(inspected);
      }
    } else {
      const discovered = discoverInstallations(baseRoot, currentScope);
      if (discovered.length > 0) {
        for (const item of discovered) {
          installations.push(inspectLayout(baseRoot, item.layout, assetsRoot, skillNames, packageInfo));
        }
      } else if (currentScope === 'local') {
        for (const layout of getLayoutRoots('local')) {
          const candidate = inspectLayout(baseRoot, layout, assetsRoot, skillNames, packageInfo);
          if (candidate.mode === 'source_linked_dev') {
            installations.push(candidate);
          }
        }
      }
    }
  }

  if (installations.length === 0) {
    issues.push('no tilelang-harness installation found; run "tilelang-harness" or "tilelang-harness init"');
  }

  const ok = issues.length === 0 && installations.every((inst) => inst.ok);
  return {
    ok,
    scope,
    workspace: localRoot,
    home_dir: globalRoot,
    package: packageInfo,
    runtime: {
      platform: process.platform,
      node: process.versions.node,
      python,
    },
    workspace_git_ignore: gitWorkspaceIgnore,
    installations,
    issues,
    warnings,
  };
}
