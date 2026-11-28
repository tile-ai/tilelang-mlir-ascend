import fs from 'node:fs';
import path from 'node:path';

export const SCOPES = Object.freeze(['global', 'local']);

export const HOSTS = Object.freeze({
  codex: Object.freeze({
    id: 'codex',
    label: 'Codex',
    aliases: Object.freeze([]),
    local: Object.freeze({
      rootDir: '.agents',
      skillsDir: '.agents/skills',
      metadataDir: '.agents/.harness',
      instructionFile: 'AGENTS.md',
    }),
    global: Object.freeze({
      rootDir: '.codex',
      skillsDir: '.codex/skills',
      metadataDir: '.codex/.harness',
      instructionFile: '.codex/AGENTS.md',
    }),
  }),
  antigravity: Object.freeze({
    id: 'antigravity',
    label: 'Antigravity',
    aliases: Object.freeze(['agy']),
    local: Object.freeze({
      rootDir: '.agents',
      skillsDir: '.agents/skills',
      metadataDir: '.agents/.harness',
      instructionFile: 'AGENTS.md',
    }),
    global: Object.freeze({
      rootDir: '.gemini/config',
      skillsDir: '.gemini/config/skills',
      metadataDir: '.gemini/config/.harness',
      instructionFile: '.gemini/GEMINI.md',
    }),
  }),
  'claude-code': Object.freeze({
    id: 'claude-code',
    label: 'Claude Code',
    aliases: Object.freeze(['claude']),
    local: Object.freeze({
      rootDir: '.claude',
      skillsDir: '.claude/skills',
      metadataDir: '.claude/.harness',
      instructionFile: 'CLAUDE.md',
    }),
    global: Object.freeze({
      rootDir: '.claude',
      skillsDir: '.claude/skills',
      metadataDir: '.claude/.harness',
      instructionFile: '.claude/CLAUDE.md',
    }),
  }),
  agent: Object.freeze({
    id: 'agent',
    label: 'Generic .agent',
    aliases: Object.freeze(['.agent', 'generic']),
    local: Object.freeze({
      rootDir: '.agent',
      skillsDir: '.agent/skills',
      metadataDir: '.agent/.harness',
      instructionFile: 'AGENTS.md',
    }),
    global: Object.freeze({
      rootDir: '.agent',
      skillsDir: '.agent/skills',
      metadataDir: '.agent/.harness',
      instructionFile: '.agent/AGENTS.md',
    }),
  }),
});

export const CANONICAL_HOST_IDS = Object.freeze(Object.keys(HOSTS));

const ALIAS_MAP = new Map();
for (const host of Object.values(HOSTS)) {
  ALIAS_MAP.set(host.id, host);
  for (const alias of host.aliases) {
    ALIAS_MAP.set(alias, host);
  }
}

export function normalizeScope(scope) {
  const s = String(scope ?? 'local').trim().toLowerCase();
  if (s !== 'global' && s !== 'local') {
    throw new Error(`unsupported install scope "${scope}"; choose "global" or "local"`);
  }
  return s;
}

export function getHostLayout(hostOrId, scope = 'local') {
  const resolvedScope = normalizeScope(scope);
  const host = typeof hostOrId === 'string' ? resolveHost(hostOrId) : hostOrId;
  if (!host || !host[resolvedScope]) {
    throw new Error(`invalid host definition: ${hostOrId}`);
  }
  const spec = host[resolvedScope];
  const skillsRefPrefix = resolvedScope === 'global' ? `~/${spec.skillsDir}` : spec.skillsDir;
  const instructionDisplay = resolvedScope === 'global' ? `~/${spec.instructionFile}` : spec.instructionFile;
  return Object.freeze({
    id: host.id,
    label: host.label,
    aliases: host.aliases,
    scope: resolvedScope,
    rootDir: spec.rootDir,
    skillsDir: spec.skillsDir,
    metadataDir: spec.metadataDir,
    instructionFile: spec.instructionFile,
    skillsRefPrefix,
    instructionDisplay,
  });
}

export function getLayoutRoots(scope = 'local') {
  const resolvedScope = normalizeScope(scope);
  const seen = new Set();
  const layouts = [];
  for (const id of CANONICAL_HOST_IDS) {
    const layout = getHostLayout(HOSTS[id], resolvedScope);
    if (!seen.has(layout.metadataDir)) {
      seen.add(layout.metadataDir);
      layouts.push(layout);
    }
  }
  return layouts;
}

export function resolveHost(name) {
  const key = String(name ?? '').trim().toLowerCase();
  const host = ALIAS_MAP.get(key);
  if (!host) {
    throw new Error(
      `unsupported host "${name}"; choose one of: ${CANONICAL_HOST_IDS.join(', ')} (aliases: agy, claude, .agent, generic)`,
    );
  }
  return host;
}

export function parseHostList(values, scope = 'local') {
  const resolvedScope = normalizeScope(scope);
  const tokens = [];
  for (const value of values) {
    for (const part of String(value).split(',')) {
      const trimmed = part.trim();
      if (trimmed) tokens.push(trimmed);
    }
  }
  const hostIds = tokens.length === 0 ? ['codex'] : tokens;
  const layouts = hostIds.map((token) => getHostLayout(resolveHost(token), resolvedScope));

  const byLayout = new Map();
  const byInstruction = new Map();
  for (const layout of layouts) {
    const layoutKey = `${layout.skillsDir}|${layout.instructionFile}`;
    const existingInstruction = byInstruction.get(layout.instructionFile);
    if (existingInstruction && existingInstruction.skillsDir !== layout.skillsDir) {
      throw new Error(
        `conflicting hosts "${existingInstruction.id}" (${existingInstruction.skillsDir}) and "${layout.id}" (${layout.skillsDir}) both manage ${layout.instructionFile}`,
      );
    }
    byInstruction.set(layout.instructionFile, layout);
    if (!byLayout.has(layoutKey)) {
      byLayout.set(layoutKey, layout);
    }
  }
  return [...byLayout.values()];
}

export function normalizeEol(text) {
  return text.replace(/\r\n/g, '\n');
}

export const EXTRA_MANAGED_SKILL_NAMES = Object.freeze([
  'ascend950-simulation',
  'tilelang-ascend-generation',
  'tilelang-ascend-knowledge',
  'ascend-roofline',
  // Retain previous skill name so managed upgrades can remove/restore its files.
  'ascend-mfu',
]);

export function isManagedSkillName(name) {
  return typeof name === 'string'
    && (name.startsWith('harness-') || EXTRA_MANAGED_SKILL_NAMES.includes(name));
}

function rewriteManagedSkillPaths(text, targetPrefix) {
  let out = text.replaceAll('.agents/skills/harness-', `${targetPrefix}/harness-`);
  for (const name of EXTRA_MANAGED_SKILL_NAMES) {
    out = out.replaceAll(`.agents/skills/${name}`, `${targetPrefix}/${name}`);
  }
  return out;
}

export function renderEntryForHost(rawEntryText, layout) {
  const normalized = normalizeEol(rawEntryText).trim();
  const targetPrefix = layout.skillsRefPrefix || layout.skillsDir;
  if (targetPrefix === '.agents/skills') {
    return normalized;
  }
  return rewriteManagedSkillPaths(normalized, targetPrefix);
}

export function renderSkillFileForHost(relativeSkillPath, buffer, layout) {
  const targetPrefix = layout.skillsRefPrefix || layout.skillsDir;
  if (targetPrefix === '.agents/skills') {
    return buffer;
  }
  if (!relativeSkillPath.endsWith('.md')) {
    return buffer;
  }
  const original = buffer.toString('utf8');
  const rewritten = rewriteManagedSkillPaths(original, targetPrefix)
    .replaceAll('`.agents/skills/`', `\`${targetPrefix}/\``)
    .replace(/(\r?\n  )\.agents\/skills\/(\r?\n)/g, `$1${targetPrefix}/$2`);
  return rewritten === original ? buffer : Buffer.from(rewritten, 'utf8');
}

export function layoutFromManifest(manifestData, fallbackLayout = getHostLayout(HOSTS.codex, 'local')) {
  const scope = manifestData && (manifestData.scope === 'global' || manifestData.scope === 'local')
    ? manifestData.scope
    : fallbackLayout.scope;
  if (manifestData && typeof manifestData.host === 'string' && ALIAS_MAP.has(manifestData.host)) {
    return getHostLayout(ALIAS_MAP.get(manifestData.host), scope);
  }
  if (manifestData && typeof manifestData.skills_dir === 'string') {
    for (const id of CANONICAL_HOST_IDS) {
      const candidate = getHostLayout(HOSTS[id], scope);
      if (candidate.skillsDir === manifestData.skills_dir) {
        return candidate;
      }
    }
  }
  return fallbackLayout;
}

export function discoverInstallations(baseRoot, scope = 'local') {
  const resolvedScope = normalizeScope(scope);
  const found = [];
  for (const layout of getLayoutRoots(resolvedScope)) {
    const manifestPath = path.join(baseRoot, ...layout.metadataDir.split('/'), 'manifest.json');
    if (fs.existsSync(manifestPath)) {
      let effectiveLayout = layout;
      try {
        const parsed = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
        effectiveLayout = layoutFromManifest(parsed, layout);
      } catch {
        // Let doctor / caller surface JSON corruption errors cleanly.
      }
      found.push({ layout: effectiveLayout, scope: resolvedScope, baseRoot, manifestPath });
    }
  }
  return found;
}
