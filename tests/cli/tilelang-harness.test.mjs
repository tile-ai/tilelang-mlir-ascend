import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  END_MARKER,
  MANIFEST_SCHEMA_V1,
  MANIFEST_SCHEMA_V2,
  SKILL_NAMES,
  START_MARKER,
  adoptLegacy,
  applyUpgrade,
  atomicWrite,
  doctorWorkspace,
  encodeCanonicalJson,
  installWorkspace,
  prepareUpgrade,
  runTransaction,
  sha256,
  uninstallWorkspace,
} from '../../src/bundle.mjs';
import { HOSTS, parseHostList, resolveHost } from '../../src/hosts.mjs';
import { main } from '../../src/cli.mjs';
import {
  applyMenuKey,
  applyScopeKey,
  createHostMenuState,
  createScopeMenuState,
  renderBanner,
  renderHostMenu,
  renderScopeMenu,
  validateSelection,
} from '../../src/prompt.mjs';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BIN_PATH = path.join(REPO_ROOT, 'bin', 'tilelang-harness.js');

function withTempDir(fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tilelang-harness-test-'));
  let result;
  try {
    result = fn(dir);
  } catch (err) {
    fs.rmSync(dir, { recursive: true, force: true });
    throw err;
  }
  if (result && typeof result.finally === 'function') {
    return result.finally(() => fs.rmSync(dir, { recursive: true, force: true }));
  }
  fs.rmSync(dir, { recursive: true, force: true });
  return result;
}

function makeFixtureBundle(root, { skillText = 'old', entryBody = 'old' } = {}) {
  const assetsRoot = path.join(root, 'assets');
  const skillDir = path.join(assetsRoot, 'skills', 'harness-example');
  fs.mkdirSync(skillDir, { recursive: true });
  fs.writeFileSync(path.join(skillDir, 'SKILL.md'), skillText, 'utf8');
  const codexDir = path.join(assetsRoot, 'codex');
  fs.mkdirSync(codexDir, { recursive: true });
  fs.writeFileSync(
    path.join(codexDir, 'AGENTS.md'),
    `${START_MARKER}\n${entryBody}\n${END_MARKER}\n`,
    'utf8',
  );
  return assetsRoot;
}

async function runCliInProcess(argv, { stdin, homeDir } = {}) {
  let out = '';
  let err = '';
  const code = await main(argv, {
    stdin: stdin ?? { isTTY: false },
    stdout: { isTTY: false, write: (chunk) => { out += chunk; } },
    stderr: { isTTY: false, write: (chunk) => { err += chunk; } },
    homeDir,
  });
  return { code, stdout: out, stderr: err };
}

test('upgrade and rollback preserve unmanaged files and user AGENTS content', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'old', entryBody: 'old' });
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    fs.writeFileSync(
      path.join(workspace, 'AGENTS.md'),
      `User instructions\n${START_MARKER}\nold\n${END_MARKER}\nTail\n`,
      'utf8',
    );

    installWorkspace({
      workspace,
      host: HOSTS.codex,
      assetsRoot,
      skillNames: ['harness-example'],
    });
    const personalPath = path.join(workspace, '.agents/skills/harness-example/personal.md');
    fs.writeFileSync(personalPath, 'keep me', 'utf8');

    // Update bundle to v2.
    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/SKILL.md'), 'new', 'utf8');
    fs.writeFileSync(
      path.join(assetsRoot, 'codex/AGENTS.md'),
      `${START_MARKER}\nnew\n${END_MARKER}\n`,
      'utf8',
    );

    const { plan } = prepareUpgrade({
      workspace,
      host: HOSTS.codex,
      assetsRoot,
      skillNames: ['harness-example'],
    });
    applyUpgrade({
      workspace,
      host: HOSTS.codex,
      planId: plan.id,
      assetsRoot,
      skillNames: ['harness-example'],
    });

    const skillPath = path.join(workspace, '.agents/skills/harness-example/SKILL.md');
    assert.equal(fs.readFileSync(skillPath, 'utf8'), 'new');
    assert.match(fs.readFileSync(path.join(workspace, 'AGENTS.md'), 'utf8'), /User instructions/);
    assert.match(fs.readFileSync(path.join(workspace, 'AGENTS.md'), 'utf8'), /Tail/);
    assert.equal(fs.readFileSync(personalPath, 'utf8'), 'keep me');

    const rolled = runTransaction({
      workspace,
      host: HOSTS.codex,
      transactionId: plan.id,
      rollback: true,
    });
    assert.equal(rolled.status, 'rolled_back');
    assert.equal(fs.readFileSync(skillPath, 'utf8'), 'old');
    assert.equal(fs.readFileSync(personalPath, 'utf8'), 'keep me');

    const { plan: nextPlan } = prepareUpgrade({
      workspace,
      host: HOSTS.codex,
      assetsRoot,
      skillNames: ['harness-example'],
    });
    assert.notEqual(nextPlan.id, plan.id);
  });
});

test('user-modified file rejects upgrade plan without modifying workspace', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'old', entryBody: 'old' });
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    installWorkspace({ workspace, assetsRoot, skillNames: ['harness-example'] });

    const skillPath = path.join(workspace, '.agents/skills/harness-example/SKILL.md');
    fs.writeFileSync(skillPath, 'user edit', 'utf8');
    const beforeAgents = fs.readFileSync(path.join(workspace, 'AGENTS.md'));

    assert.throws(
      () => prepareUpgrade({ workspace, assetsRoot, skillNames: ['harness-example'] }),
      /user-modified/,
    );
    assert.deepEqual(fs.readFileSync(path.join(workspace, 'AGENTS.md')), beforeAgents);
  });
});

test('interrupted upgrade resumes cleanly without requiring source bundle', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'old', entryBody: 'old' });
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    installWorkspace({ workspace, assetsRoot, skillNames: ['harness-example'] });

    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/SKILL.md'), 'new', 'utf8');
    fs.writeFileSync(
      path.join(assetsRoot, 'codex/AGENTS.md'),
      `${START_MARKER}\nnew\n${END_MARKER}\n`,
      'utf8',
    );

    const { plan } = prepareUpgrade({ workspace, assetsRoot, skillNames: ['harness-example'] });
    const agentsPath = path.join(workspace, 'AGENTS.md');
    const failingWriter = (filePath, data) => {
      if (path.resolve(filePath) === path.resolve(agentsPath)) {
        throw new Error('synthetic interruption');
      }
      return atomicWrite(filePath, data);
    };

    assert.throws(
      () => applyUpgrade({
        workspace,
        planId: plan.id,
        assetsRoot,
        skillNames: ['harness-example'],
        atomicWriter: failingWriter,
      }),
      /synthetic interruption/,
    );
    assert.equal(
      fs.readFileSync(path.join(workspace, '.agents/skills/harness-example/SKILL.md'), 'utf8'),
      'new',
    );

    // Doctor should report the interrupted transaction.
    const report = doctorWorkspace({ workspace, assetsRoot, skillNames: ['harness-example'] });
    assert.equal(report.ok, false);
    assert.ok(report.installations[0].pending_transactions.length === 1);

    // Resume without source bundle.
    fs.rmSync(assetsRoot, { recursive: true, force: true });
    const resumed = runTransaction({ workspace, transactionId: plan.id });
    assert.equal(resumed.status, 'completed');
    assert.match(fs.readFileSync(agentsPath, 'utf8'), /\nnew\n/);
  });
});

test('interrupted rollback resumes in rollback direction', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'old', entryBody: 'old' });
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    installWorkspace({ workspace, assetsRoot, skillNames: ['harness-example'] });

    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/SKILL.md'), 'new', 'utf8');
    fs.writeFileSync(
      path.join(assetsRoot, 'codex/AGENTS.md'),
      `${START_MARKER}\nnew\n${END_MARKER}\n`,
      'utf8',
    );

    const { plan } = prepareUpgrade({ workspace, assetsRoot, skillNames: ['harness-example'] });
    applyUpgrade({ workspace, planId: plan.id, assetsRoot, skillNames: ['harness-example'] });

    const agentsPath = path.join(workspace, 'AGENTS.md');
    const failingWriter = (filePath, data) => {
      if (path.resolve(filePath) === path.resolve(agentsPath)) {
        throw new Error('synthetic rollback interruption');
      }
      return atomicWrite(filePath, data);
    };

    assert.throws(
      () => runTransaction({
        workspace,
        transactionId: plan.id,
        rollback: true,
        atomicWriter: failingWriter,
      }),
      /synthetic rollback interruption/,
    );
    assert.equal(
      fs.readFileSync(path.join(workspace, '.agents/skills/harness-example/SKILL.md'), 'utf8'),
      'old',
    );

    // Resuming without rollback flag continues in rollback direction.
    const resumed = runTransaction({ workspace, transactionId: plan.id });
    assert.equal(resumed.status, 'rolled_back');
    assert.match(fs.readFileSync(agentsPath, 'utf8'), /\nold\n/);
  });
});

test('rollback refuses to overwrite post-upgrade user edits', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'old', entryBody: 'old' });
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    installWorkspace({ workspace, assetsRoot, skillNames: ['harness-example'] });

    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/SKILL.md'), 'new', 'utf8');
    fs.writeFileSync(
      path.join(assetsRoot, 'codex/AGENTS.md'),
      `${START_MARKER}\nnew\n${END_MARKER}\n`,
      'utf8',
    );
    const { plan } = prepareUpgrade({ workspace, assetsRoot, skillNames: ['harness-example'] });
    applyUpgrade({ workspace, planId: plan.id, assetsRoot, skillNames: ['harness-example'] });

    fs.writeFileSync(path.join(workspace, '.agents/skills/harness-example/SKILL.md'), 'new user edit', 'utf8');
    assert.throws(
      () => runTransaction({ workspace, transactionId: plan.id, rollback: true }),
      /changed after/,
    );
    assert.match(fs.readFileSync(path.join(workspace, 'AGENTS.md'), 'utf8'), /\nnew\n/);
  });
});

test('changed source bundle invalidates reviewed upgrade plan', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'old', entryBody: 'old' });
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    installWorkspace({ workspace, assetsRoot, skillNames: ['harness-example'] });

    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/SKILL.md'), 'new', 'utf8');
    const { plan } = prepareUpgrade({ workspace, assetsRoot, skillNames: ['harness-example'] });

    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/SKILL.md'), 'different', 'utf8');
    assert.throws(
      () => applyUpgrade({ workspace, planId: plan.id, assetsRoot, skillNames: ['harness-example'] }),
      /plan changed/,
    );
    assert.equal(
      fs.readFileSync(path.join(workspace, '.agents/skills/harness-example/SKILL.md'), 'utf8'),
      'old',
    );
  });
});

test('removed managed file is deleted during upgrade while extra user file survives', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'old', entryBody: 'old' });
    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/obsolete.md'), 'gone soon', 'utf8');
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    installWorkspace({ workspace, assetsRoot, skillNames: ['harness-example'] });

    const personal = path.join(workspace, '.agents/skills/harness-example/personal.md');
    fs.writeFileSync(personal, 'keep me', 'utf8');
    fs.unlinkSync(path.join(assetsRoot, 'skills/harness-example/obsolete.md'));

    const { plan } = prepareUpgrade({ workspace, assetsRoot, skillNames: ['harness-example'] });
    applyUpgrade({ workspace, planId: plan.id, assetsRoot, skillNames: ['harness-example'] });

    assert.equal(fs.existsSync(path.join(workspace, '.agents/skills/harness-example/obsolete.md')), false);
    assert.equal(fs.readFileSync(personal, 'utf8'), 'keep me');
  });
});

test('source-linked dev workspace rejects install before writing manifest', () => {
  withTempDir((tmp) => {
    const linked = path.join(tmp, 'linked');
    const target = path.join(linked, '.agents/skills');
    fs.mkdirSync(target, { recursive: true });
    fs.symlinkSync(
      path.join(REPO_ROOT, 'assets/skills', SKILL_NAMES[0]),
      path.join(target, SKILL_NAMES[0]),
      'dir',
    );
    assert.throws(
      () => installWorkspace({ workspace: linked, host: HOSTS.codex }),
      /source-linked/,
    );
    assert.equal(fs.existsSync(path.join(linked, '.agents/.harness/manifest.json')), false);

    // Doctor recognizes source-linked dev mode as healthy info.
    const report = doctorWorkspace({ workspace: linked });
    assert.equal(report.ok, true);
    assert.equal(report.installations[0].mode, 'source_linked_dev');
  });
});

test('legacy adoption requires exact match of old bundle and enables upgrades', () => {
  withTempDir((tmp) => {
    const legacy = path.join(tmp, 'legacy');
    makeFixtureBundle(legacy, { skillText: 'wrong', entryBody: 'old' });

    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(path.join(workspace, '.agents/skills/harness-example'), { recursive: true });
    fs.writeFileSync(path.join(workspace, '.agents/skills/harness-example/SKILL.md'), 'old', 'utf8');
    fs.writeFileSync(
      path.join(workspace, 'AGENTS.md'),
      `Header\n${START_MARKER}\nold\n${END_MARKER}\n`,
      'utf8',
    );

    assert.throws(
      () => adoptLegacy({ workspace, bundleRoot: legacy }),
      /does not match/,
    );
    assert.equal(fs.existsSync(path.join(workspace, '.agents/.harness/manifest.json')), false);

    fs.writeFileSync(path.join(legacy, 'assets/skills/harness-example/SKILL.md'), 'old', 'utf8');
    const adopted = adoptLegacy({ workspace, bundleRoot: legacy });
    assert.equal(adopted.status, 'adopted');
    assert.equal(adopted.files, 1);

    // Now upgrade to a newer bundle.
    const newer = path.join(tmp, 'newer');
    const newerAssets = makeFixtureBundle(newer, { skillText: 'new', entryBody: 'new' });
    const { plan } = prepareUpgrade({ workspace, assetsRoot: newerAssets, skillNames: ['harness-example'] });
    applyUpgrade({ workspace, planId: plan.id, assetsRoot: newerAssets, skillNames: ['harness-example'] });
    assert.equal(
      fs.readFileSync(path.join(workspace, '.agents/skills/harness-example/SKILL.md'), 'utf8'),
      'new',
    );
  });
});

test('supports all 4 host targets, aliases, path rewriting, and coexistence', async () => {
  assert.equal(resolveHost('agy').id, 'antigravity');
  assert.equal(resolveHost('claude').id, 'claude-code');
  assert.equal(resolveHost('.agent').id, 'agent');
  assert.equal(resolveHost('generic').id, 'agent');
  assert.throws(() => parseHostList(['codex,agent']), /conflicting hosts/);

  // 1. Codex + Claude Code coexist cleanly in the same workspace.
  await withTempDir(async (tmp) => {
    const res = await runCliInProcess(['init', '--workspace', tmp, '--host', 'codex,claude-code', '--json']);
    assert.equal(res.code, 0, res.stderr);
    const parsed = JSON.parse(res.stdout);
    assert.equal(parsed.length, 2);

    assert.equal(fs.readdirSync(path.join(tmp, '.agents/skills')).length, SKILL_NAMES.length);
    assert.equal(fs.readdirSync(path.join(tmp, '.claude/skills')).length, SKILL_NAMES.length);

    // Claude Code entry and skill docs rewrite .agents/skills/harness-* -> .claude/skills/harness-*
    const claudeMd = fs.readFileSync(path.join(tmp, 'CLAUDE.md'), 'utf8');
    assert.match(claudeMd, /\.claude\/skills\/harness-new-project\/references\/project-management\.md/);
    assert.doesNotMatch(claudeMd, /\.agents\/skills\/harness-/);

    const claudeSearchDoc = fs.readFileSync(
      path.join(tmp, '.claude/skills/harness-resume/references/search-knowledge.md'),
      'utf8',
    );
    assert.match(claudeSearchDoc, /\.claude\/skills\/harness-resume\/scripts\/searchctl\.py/);
    assert.doesNotMatch(claudeSearchDoc, /\.agents\/skills\/harness-/);

    // External TileLang repo reference must NOT be rewritten.
    const claudeWorkflowDoc = fs.readFileSync(
      path.join(tmp, '.claude/skills/harness-develop/references/tilelang-workflow.md'),
      'utf8',
    );
    assert.match(claudeWorkflowDoc, /\.agents\/skills\/tilelang-ascend\/SKILL\.md/);

    // Doctor reports both installations healthy.
    const doc = await runCliInProcess(['doctor', '--workspace', tmp, '--json']);
    assert.equal(doc.code, 0, doc.stderr);
    const report = JSON.parse(doc.stdout);
    assert.equal(report.ok, true);
    assert.equal(report.installations.length, 2);

    // Ambiguous uninstall without --host fails safely when two layouts exist.
    const ambig = await runCliInProcess(['uninstall', '--workspace', tmp]);
    assert.notEqual(ambig.code, 0);
    assert.match(ambig.stderr, /multiple local installations detected/);
  });

  // 2. Antigravity and generic .agent each install cleanly.
  await withTempDir(async (tmp) => {
    const res = await runCliInProcess(['init', '--workspace', tmp, '--host', 'antigravity']);
    assert.equal(res.code, 0, res.stderr);
    const manifest = JSON.parse(fs.readFileSync(path.join(tmp, '.agents/.harness/manifest.json'), 'utf8'));
    assert.equal(manifest.schema, MANIFEST_SCHEMA_V2);
    assert.equal(manifest.host, 'antigravity');
    assert.equal(manifest.skills_dir, '.agents/skills');
  });

  await withTempDir(async (tmp) => {
    const res = await runCliInProcess(['init', '--workspace', tmp, '--host', '.agent']);
    assert.equal(res.code, 0, res.stderr);
    assert.equal(fs.readdirSync(path.join(tmp, '.agent/skills')).length, SKILL_NAMES.length);
    const agentsMd = fs.readFileSync(path.join(tmp, 'AGENTS.md'), 'utf8');
    assert.match(agentsMd, /\.agent\/skills\/harness-new-project\/references\/project-management\.md/);
    const manifest = JSON.parse(fs.readFileSync(path.join(tmp, '.agent/.harness/manifest.json'), 'utf8'));
    assert.equal(manifest.host, 'agent');
    assert.equal(manifest.skills_dir, '.agent/skills');
  });
});

test('seamlessly upgrades legacy installation@1 manifest and tolerates CRLF in AGENTS.md', () => {
  withTempDir((tmp) => {
    const assetsRoot = makeFixtureBundle(tmp, { skillText: 'v1', entryBody: 'entry-v1' });
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(path.join(workspace, '.agents/skills/harness-example'), { recursive: true });
    fs.mkdirSync(path.join(workspace, '.agents/.harness'), { recursive: true });

    const skillPath = path.join(workspace, '.agents/skills/harness-example/SKILL.md');
    fs.writeFileSync(skillPath, 'v1', 'utf8');
    // Simulate Windows CRLF in AGENTS.md.
    fs.writeFileSync(
      path.join(workspace, 'AGENTS.md'),
      `# My Notes\r\n\r\n${START_MARKER}\r\nentry-v1\r\n${END_MARKER}\r\n`,
      'utf8',
    );
    const v1Manifest = {
      schema: MANIFEST_SCHEMA_V1,
      files: { '.agents/skills/harness-example/SKILL.md': sha256(Buffer.from('v1', 'utf8')) },
      entry_sha256: sha256(Buffer.from(`${START_MARKER}\nentry-v1\n${END_MARKER}`, 'utf8')),
    };
    fs.writeFileSync(
      path.join(workspace, '.agents/.harness/manifest.json'),
      encodeCanonicalJson(v1Manifest),
    );

    // Upgrade to v2 bundle.
    fs.writeFileSync(path.join(assetsRoot, 'skills/harness-example/SKILL.md'), 'v2', 'utf8');
    fs.writeFileSync(
      path.join(assetsRoot, 'codex/AGENTS.md'),
      `${START_MARKER}\nentry-v2\n${END_MARKER}\n`,
      'utf8',
    );

    const { plan } = prepareUpgrade({ workspace, assetsRoot, skillNames: ['harness-example'] });
    applyUpgrade({ workspace, planId: plan.id, assetsRoot, skillNames: ['harness-example'] });

    assert.equal(fs.readFileSync(skillPath, 'utf8'), 'v2');
    const updatedAgents = fs.readFileSync(path.join(workspace, 'AGENTS.md'), 'utf8');
    assert.match(updatedAgents, /# My Notes\r\n/);
    assert.match(updatedAgents, /entry-v2\r\n/);
    const upgradedManifest = JSON.parse(
      fs.readFileSync(path.join(workspace, '.agents/.harness/manifest.json'), 'utf8'),
    );
    assert.equal(upgradedManifest.schema, MANIFEST_SCHEMA_V2);
  });
});

test('uninstall removes only managed files and Harness block, preserving user data and .workspace/', () => {
  withTempDir((tmp) => {
    const workspace = path.join(tmp, 'workspace');
    fs.mkdirSync(workspace);
    fs.writeFileSync(path.join(workspace, 'CLAUDE.md'), '# Project Rules\nKeep this section.\n', 'utf8');
    fs.mkdirSync(path.join(workspace, '.workspace/my-op'), { recursive: true });
    fs.writeFileSync(path.join(workspace, '.workspace/my-op/STATE.md'), 'active work', 'utf8');

    installWorkspace({ workspace, host: HOSTS['claude-code'] });
    const extraUserFile = path.join(workspace, '.claude/skills/harness-develop/my-notes.md');
    fs.writeFileSync(extraUserFile, 'custom notes', 'utf8');

    // Dry-run makes no changes.
    const dry = uninstallWorkspace({ workspace, host: HOSTS['claude-code'], dryRun: true });
    assert.equal(dry.status, 'dry_run');
    assert.ok(dry.files_to_remove.length > 10);
    assert.equal(dry.instruction_action, 'strip_block');
    assert.equal(fs.existsSync(path.join(workspace, '.claude/.harness/manifest.json')), true);

    // Modifying a managed file blocks uninstall unless --force.
    const managedSkill = path.join(workspace, '.claude/skills/harness-status/SKILL.md');
    const originalSkill = fs.readFileSync(managedSkill);
    fs.writeFileSync(managedSkill, 'edited by user', 'utf8');
    assert.throws(
      () => uninstallWorkspace({ workspace, host: HOSTS['claude-code'] }),
      /user-modified managed file/,
    );
    fs.writeFileSync(managedSkill, originalSkill);

    // Clean uninstall succeeds.
    const res = uninstallWorkspace({ workspace, host: HOSTS['claude-code'] });
    assert.equal(res.status, 'uninstalled');
    assert.equal(fs.existsSync(managedSkill), false);
    assert.equal(fs.existsSync(path.join(workspace, '.claude/.harness/manifest.json')), false);
    // Preserved: extra user file, user CLAUDE.md section, and .workspace/.
    assert.equal(fs.readFileSync(extraUserFile, 'utf8'), 'custom notes');
    assert.equal(fs.readFileSync(path.join(workspace, 'CLAUDE.md'), 'utf8'), '# Project Rules\nKeep this section.\n');
    assert.equal(fs.readFileSync(path.join(workspace, '.workspace/my-op/STATE.md'), 'utf8'), 'active work');
  });

  // Uninstall deletes instruction file when it only contained the Harness block.
  withTempDir((tmp) => {
    installWorkspace({ workspace: tmp, host: HOSTS.codex });
    const res = uninstallWorkspace({ workspace: tmp, host: HOSTS.codex });
    assert.equal(res.instruction_action, 'delete');
    assert.equal(fs.existsSync(path.join(tmp, 'AGENTS.md')), false);
    assert.equal(fs.existsSync(path.join(tmp, '.agents')), false);
  });
});

test('packed npm tarball initializes, doctors, and uninstalls in an isolated directory without source repo', () => {
  withTempDir((tmp) => {
    const packDir = path.join(tmp, 'pack');
    const unpackDir = path.join(tmp, 'unpacked');
    const workspace = path.join(tmp, 'user-project');
    fs.mkdirSync(packDir);
    fs.mkdirSync(unpackDir);
    fs.mkdirSync(workspace);

    const packRes = spawnSync('npm', ['pack', '--pack-destination', packDir], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    });
    assert.equal(packRes.status, 0, packRes.stderr);
    const tgzName = fs.readdirSync(packDir).find((f) => f.endsWith('.tgz'));
    assert.ok(tgzName, 'expected npm pack tarball');

    const tarRes = spawnSync('tar', ['-xzf', path.join(packDir, tgzName), '-C', unpackDir], {
      encoding: 'utf8',
    });
    assert.equal(tarRes.status, 0, tarRes.stderr);

    const packedBin = path.join(unpackDir, 'package', 'bin', 'tilelang-harness.js');
    assert.equal(fs.existsSync(packedBin), true);

    const initRes = spawnSync(
      process.execPath,
      [packedBin, 'init', '--workspace', workspace, '--host', 'claude-code'],
      { cwd: workspace, encoding: 'utf8' },
    );
    assert.equal(initRes.status, 0, initRes.stderr);
    assert.equal(fs.readdirSync(path.join(workspace, '.claude/skills')).length, SKILL_NAMES.length);

    const doctorRes = spawnSync(
      process.execPath,
      [packedBin, 'doctor', '--workspace', workspace, '--json'],
      { cwd: workspace, encoding: 'utf8' },
    );
    assert.equal(doctorRes.status, 0, doctorRes.stderr);
    const report = JSON.parse(doctorRes.stdout);
    assert.equal(report.ok, true);
    assert.equal(report.installations[0].host, 'claude-code');

    const uninstallRes = spawnSync(
      process.execPath,
      [packedBin, 'uninstall', '--workspace', workspace],
      { cwd: workspace, encoding: 'utf8' },
    );
    assert.equal(uninstallRes.status, 0, uninstallRes.stderr);
    assert.equal(fs.existsSync(path.join(workspace, '.claude')), false);
  });
});

test('bin executable runs cleanly via subprocess', () => {
  const res = spawnSync(process.execPath, [BIN_PATH, '--version'], { encoding: 'utf8' });
  assert.equal(res.status, 0, res.stderr);
  assert.match(res.stdout.trim(), /^\d+\.\d+\.\d+/);
});

test('interactive menu state machine handles scope, navigation, multi-select, conflicts, and cancel', () => {
  const banner = renderBanner({ name: 'tilelang-harness', version: '0.1.0' }, { isTTY: false });
  assert.match(banner, /TileLang Harness v0\.1\.0/);
  assert.match(banner, /╭/);

  // Scope menu defaults to global and toggles to local.
  let scopeState = createScopeMenuState('global');
  assert.match(renderScopeMenu(scopeState, { isTTY: false }).join('\n'), /Global/);
  scopeState = applyScopeKey(scopeState, { name: 'down' });
  scopeState = applyScopeKey(scopeState, { name: 'return' });
  assert.equal(scopeState.done, true);
  assert.equal(scopeState.items[scopeState.cursor].id, 'local');

  // Local mode host menu rejects conflicting AGENTS.md layouts.
  let state = createHostMenuState(['codex'], 'local');
  state = applyMenuKey(state, { name: 'space', sequence: ' ' });
  state = applyMenuKey(state, { name: 'return' });
  assert.equal(state.done, false);
  assert.match(state.error, /Select at least one host/);

  state = applyMenuKey(state, { sequence: '1' });
  state = applyMenuKey(state, { sequence: '4' });
  state = applyMenuKey(state, { name: 'return' });
  assert.equal(state.done, false);
  assert.match(state.error, /conflicting hosts/);

  state = applyMenuKey(state, { sequence: '4' });
  state = applyMenuKey(state, { name: 'up' });
  state = applyMenuKey(state, { name: 'space', sequence: ' ' });
  const rendered = renderHostMenu(state, { isTTY: false }).join('\n');
  assert.match(rendered, /◉ Codex/);
  assert.match(rendered, /◉ Claude Code/);
  state = applyMenuKey(state, { name: 'return' });
  assert.equal(state.done, true);
  assert.equal(state.cancelled, false);
  assert.deepEqual([...state.selected].sort(), ['claude-code', 'codex']);

  // Global mode allows ALL 4 hosts simultaneously with zero conflicts.
  const allGlobal = validateSelection(['codex', 'antigravity', 'claude-code', 'agent'], 'global');
  assert.equal(allGlobal.valid, true);
  assert.equal(allGlobal.error, null);

  // Local mode shared-layout informational note.
  const shared = validateSelection(['codex', 'antigravity'], 'local');
  assert.equal(shared.valid, true);
  assert.match(shared.note, /share the same/);

  // Cancel via q.
  const cancelled = applyMenuKey(createHostMenuState(['codex'], 'global'), { sequence: 'q' });
  assert.equal(cancelled.done, true);
  assert.equal(cancelled.cancelled, true);
});

test('global install supports all 4 hosts simultaneously, rewrites ~/ paths, and uninstalls cleanly', async () => {
  await withTempDir(async (fakeHome) => {
    fs.mkdirSync(path.join(fakeHome, '.codex'), { recursive: true });
    fs.writeFileSync(path.join(fakeHome, '.codex/config.toml'), 'model = "o3"\n', 'utf8');

    const initRes = await runCliInProcess(
      ['init', '--global', '--host', 'codex,antigravity,claude-code,agent', '--json'],
      { homeDir: fakeHome },
    );
    assert.equal(initRes.code, 0, initRes.stderr);
    const results = JSON.parse(initRes.stdout);
    assert.equal(results.length, 4);

    // All 4 global skill dirs & instruction files installed cleanly.
    assert.equal(fs.readdirSync(path.join(fakeHome, '.codex/skills')).length, SKILL_NAMES.length);
    assert.equal(fs.readdirSync(path.join(fakeHome, '.gemini/config/skills')).length, SKILL_NAMES.length);
    assert.equal(fs.readdirSync(path.join(fakeHome, '.claude/skills')).length, SKILL_NAMES.length);
    assert.equal(fs.readdirSync(path.join(fakeHome, '.agent/skills')).length, SKILL_NAMES.length);

    assert.match(
      fs.readFileSync(path.join(fakeHome, '.codex/AGENTS.md'), 'utf8'),
      /~\/\.codex\/skills\/harness-new-project\/references\/project-management\.md/,
    );
    assert.match(
      fs.readFileSync(path.join(fakeHome, '.gemini/GEMINI.md'), 'utf8'),
      /~\/\.gemini\/config\/skills\/harness-new-project\/references\/project-management\.md/,
    );
    assert.match(
      fs.readFileSync(path.join(fakeHome, '.claude/CLAUDE.md'), 'utf8'),
      /~\/\.claude\/skills\/harness-new-project\/references\/project-management\.md/,
    );
    assert.match(
      fs.readFileSync(path.join(fakeHome, '.agent/AGENTS.md'), 'utf8'),
      /~\/\.agent\/skills\/harness-new-project\/references\/project-management\.md/,
    );

    // Bundled script docs point to ~/ global skills while keeping external TileLang references untouched.
    const codexSearchDoc = fs.readFileSync(
      path.join(fakeHome, '.codex/skills/harness-resume/references/search-knowledge.md'),
      'utf8',
    );
    assert.match(codexSearchDoc, /~\/\.codex\/skills\/harness-resume\/scripts\/searchctl\.py/);
    const codexWorkflowDoc = fs.readFileSync(
      path.join(fakeHome, '.codex/skills/harness-develop/references/tilelang-workflow.md'),
      'utf8',
    );
    assert.match(codexWorkflowDoc, /\.agents\/skills\/tilelang-ascend\/SKILL\.md/);

    // Global doctor verifies all 4 installations healthy.
    const docRes = await runCliInProcess(['doctor', '--global', '--json'], { homeDir: fakeHome });
    assert.equal(docRes.code, 0, docRes.stderr);
    const report = JSON.parse(docRes.stdout);
    assert.equal(report.ok, true);
    assert.equal(report.installations.length, 4);

    // Global uninstall removes managed files/manifest but NEVER removes ~/.codex or user config.toml.
    const unRes = await runCliInProcess(
      ['uninstall', '--global', '--host', 'codex', '--json'],
      { homeDir: fakeHome },
    );
    assert.equal(unRes.code, 0, unRes.stderr);
    assert.equal(fs.existsSync(path.join(fakeHome, '.codex/skills')), false);
    assert.equal(fs.existsSync(path.join(fakeHome, '.codex/AGENTS.md')), false);
    assert.equal(fs.readFileSync(path.join(fakeHome, '.codex/config.toml'), 'utf8'), 'model = "o3"\n');
  });
});

test('local install does not mutate .gitignore and doctor handles normal repo and linked git worktree', async () => {
  await withTempDir(async (tmp) => {
    const repo = path.join(tmp, 'main-repo');
    const worktree = path.join(tmp, 'linked-wt');
    assert.equal(spawnSync('git', ['init', '-q', repo], { encoding: 'utf8' }).status, 0);
    assert.equal(
      spawnSync(
        'git',
        ['-C', repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '--allow-empty', '-m', 'init', '-q'],
        { encoding: 'utf8' },
      ).status,
      0,
    );
    assert.equal(
      spawnSync('git', ['-C', repo, 'worktree', 'add', '-q', worktree, '-b', 'wt-branch'], { encoding: 'utf8' }).status,
      0,
    );
    // Confirm linked worktree has a .git file (not a directory).
    assert.equal(fs.statSync(path.join(worktree, '.git')).isFile(), true);

    const initMain = await runCliInProcess(['init', '--local', '--dir', repo, '--host', 'codex']);
    assert.equal(initMain.code, 0, initMain.stderr);
    const initWt = await runCliInProcess(['init', '--local', '--dir', worktree, '--host', 'codex']);
    assert.equal(initWt.code, 0, initWt.stderr);
    assert.equal(fs.existsSync(path.join(repo, '.gitignore')), false);
    assert.equal(fs.existsSync(path.join(worktree, '.gitignore')), false);

    // Create .workspace/ in both main checkout and linked worktree without ignoring it.
    fs.mkdirSync(path.join(repo, '.workspace/demo'), { recursive: true });
    fs.mkdirSync(path.join(worktree, '.workspace/demo'), { recursive: true });

    const wtWarnDoc = await runCliInProcess(['doctor', '--local', '--dir', worktree, '--json']);
    assert.equal(wtWarnDoc.code, 0, wtWarnDoc.stderr);
    const wtWarnReport = JSON.parse(wtWarnDoc.stdout);
    assert.equal(wtWarnReport.workspace_git_ignore.is_git_repo, true);
    assert.equal(wtWarnReport.workspace_git_ignore.ignored, false);
    const excludePath = wtWarnReport.workspace_git_ignore.exclude_path;
    assert.ok(excludePath && excludePath.endsWith(path.join('info', 'exclude')));
    assert.equal(fs.realpathSync(excludePath), fs.realpathSync(path.join(repo, '.git', 'info', 'exclude')));
    assert.ok(wtWarnReport.warnings.some((w) => w.includes(excludePath)));

    // Append .workspace/ to the resolved common info/exclude -> both main checkout and linked worktree are now ignored!
    fs.appendFileSync(excludePath, '\n.workspace/\n', 'utf8');

    for (const dir of [repo, worktree]) {
      const cleanDoc = await runCliInProcess(['doctor', '--local', '--dir', dir, '--json']);
      assert.equal(cleanDoc.code, 0, cleanDoc.stderr);
      const cleanReport = JSON.parse(cleanDoc.stdout);
      assert.equal(cleanReport.workspace_git_ignore.ignored, true);
      assert.equal(cleanReport.warnings.some((w) => w.includes('.workspace/')), false);
      assert.equal(fs.existsSync(path.join(dir, '.gitignore')), false);
    }
  });
});

test('interactive TTY bare wizard defaults to global scope and installs without asking workspace dir', async () => {
  await withTempDir(async (fakeHome) => {
    const stdin = new PassThrough();
    stdin.isTTY = true;
    stdin.isRaw = false;
    stdin.setRawMode = (mode) => { stdin.isRaw = mode; };

    let stdoutText = '';
    const stdout = new PassThrough();
    stdout.isTTY = true;
    stdout.on('data', (chunk) => { stdoutText += chunk.toString('utf8'); });

    let stderrText = '';
    const stderr = { isTTY: true, write: (chunk) => { stderrText += chunk; } };

    // Bare invocation `npx tilelang-harness`:
    // Step 1 (scope): Enter -> Global
    // Step 2 (hosts): down, down (Claude Code), Space (select alongside Codex), Enter
    const runPromise = main([], { stdin, stdout, stderr, homeDir: fakeHome });
    setImmediate(() => {
      stdin.write('\r'); // confirm Global scope
      setImmediate(() => {
        stdin.write('\x1b[B'); // down -> antigravity
        stdin.write('\x1b[B'); // down -> claude-code
        stdin.write(' ');      // toggle claude-code
        stdin.write('\r');     // confirm hosts
      });
    });
    const code = await runPromise;
    assert.equal(code, 0, stderrText);
    assert.match(stdoutText, /TileLang Harness/);
    assert.match(stdoutText, /Scope:.*Global/);
    assert.match(stdoutText, /Hosts:.*Codex, Claude Code/);
    assert.equal(fs.readdirSync(path.join(fakeHome, '.codex/skills')).length, SKILL_NAMES.length);
    assert.equal(fs.readdirSync(path.join(fakeHome, '.claude/skills')).length, SKILL_NAMES.length);
  });
});

test('interactive TTY cancellation exits cleanly without writing files', async () => {
  await withTempDir(async (tmp) => {
    const stdin = new PassThrough();
    stdin.isTTY = true;
    stdin.isRaw = false;
    stdin.setRawMode = (mode) => { stdin.isRaw = mode; };
    const stdout = new PassThrough();
    stdout.isTTY = true;
    const stderr = { isTTY: true, write: () => {} };

    const runPromise = main(['init', '--local', '--dir', tmp], { stdin, stdout, stderr });
    setImmediate(() => {
      stdin.write('q');
    });
    const code = await runPromise;
    assert.equal(code, 130);
    assert.equal(fs.existsSync(path.join(tmp, '.agents')), false);
  });
});

test('non-TTY bare invocation prints help and never hangs', async () => {
  const res = await runCliInProcess([]);
  assert.equal(res.code, 0);
  assert.match(res.stdout, /Usage:/);
  assert.match(res.stdout, /Launch interactive installer/);
});
