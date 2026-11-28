import path from 'node:path';
import readline from 'node:readline';
import { CANONICAL_HOST_IDS, HOSTS, getHostLayout, normalizeScope, parseHostList } from './hosts.mjs';

function makeColors(enabled) {
  const wrap = (code) => (text) => (enabled ? `\x1b[${code}m${text}\x1b[0m` : text);
  return {
    bold: wrap('1'),
    dim: wrap('2'),
    red: wrap('31'),
    green: wrap('32'),
    yellow: wrap('33'),
    cyan: wrap('36'),
  };
}

function colorsEnabled(stdout) {
  return Boolean(stdout && stdout.isTTY && !process.env.NO_COLOR);
}

export function renderBanner(pkg, stdout = process.stdout) {
  const c = makeColors(colorsEnabled(stdout));
  const title = `TileLang Harness v${pkg.version}`;
  const subtitle = 'Ascend NPU TileLang Operator Harness Installer';
  const width = Math.max(title.length, subtitle.length) + 4;
  const top = `╭${'─'.repeat(width)}╮`;
  const bottom = `╰${'─'.repeat(width)}╯`;
  const padLine = (raw, styled) => {
    const rightPad = ' '.repeat(Math.max(0, width - 2 - raw.length));
    return `│  ${styled}${rightPad}│`;
  };
  return [
    c.cyan(top),
    padLine(title, c.bold(c.cyan(title))),
    padLine(subtitle, c.dim(subtitle)),
    c.cyan(bottom),
  ].join('\n');
}

export const SCOPE_OPTIONS = Object.freeze([
  Object.freeze({
    id: 'global',
    label: 'Global',
    hint: 'Install once for all repos (~/.codex, ~/.gemini, ~/.claude, ~/.agent)',
  }),
  Object.freeze({
    id: 'local',
    label: 'Local',
    hint: 'Install into a single project directory (./)',
  }),
]);

export function createScopeMenuState(defaultScope = 'global') {
  const idx = SCOPE_OPTIONS.findIndex((o) => o.id === defaultScope);
  return {
    items: SCOPE_OPTIONS,
    cursor: idx >= 0 ? idx : 0,
    done: false,
    cancelled: false,
  };
}

export function applyScopeKey(state, key = {}) {
  const { name, ctrl, sequence } = key;
  const total = state.items.length;

  if ((ctrl && name === 'c') || name === 'escape' || sequence === 'q' || sequence === 'Q') {
    return { ...state, cancelled: true, done: true };
  }
  if (name === 'up' || sequence === 'k' || sequence === 'K') {
    return { ...state, cursor: (state.cursor - 1 + total) % total };
  }
  if (name === 'down' || sequence === 'j' || sequence === 'J') {
    return { ...state, cursor: (state.cursor + 1) % total };
  }
  if (sequence === '1') {
    return { ...state, cursor: 0 };
  }
  if (sequence === '2') {
    return { ...state, cursor: 1 };
  }
  if (name === 'return' || name === 'enter' || name === 'space' || sequence === ' ') {
    return { ...state, done: true };
  }
  return state;
}

export function renderScopeMenu(state, stdout = process.stdout) {
  const c = makeColors(colorsEnabled(stdout));
  const lines = [];
  lines.push(`${c.cyan('?')} ${c.bold('Where would you like to install TileLang Harness?')}`);
  lines.push(c.dim('  (↑/↓ move, Enter confirm, q cancel)'));
  lines.push('');

  state.items.forEach((opt, idx) => {
    const focused = idx === state.cursor;
    const pointer = focused ? c.cyan('❯') : ' ';
    const radio = focused ? c.green('●') : c.dim('○');
    const label = focused ? c.bold(opt.label.padEnd(8)) : opt.label.padEnd(8);
    lines.push(`  ${pointer} ${radio} ${label} ${c.dim(opt.hint)}`);
  });
  return lines;
}

export function validateSelection(selectedIds, scope = 'global') {
  const resolvedScope = normalizeScope(scope);
  if (!selectedIds || selectedIds.length === 0) {
    return {
      valid: false,
      error: 'Select at least one host (press Space to toggle).',
      note: null,
    };
  }
  try {
    parseHostList(selectedIds, resolvedScope);
  } catch (err) {
    return {
      valid: false,
      error: err instanceof Error ? err.message : String(err),
      note: null,
    };
  }
  const hasCodex = selectedIds.includes('codex');
  const hasAntigravity = selectedIds.includes('antigravity');
  const note = resolvedScope === 'local' && hasCodex && hasAntigravity
    ? 'In local mode, Codex and Antigravity share the same .agents/skills + AGENTS.md layout.'
    : null;
  return { valid: true, error: null, note };
}

export function createHostMenuState(initialSelected = ['codex'], scope = 'global') {
  const resolvedScope = normalizeScope(scope);
  const items = CANONICAL_HOST_IDS.map((id) => getHostLayout(HOSTS[id], resolvedScope));
  const selected = new Set(initialSelected);
  return {
    scope: resolvedScope,
    items,
    cursor: 0,
    selected,
    error: null,
    done: false,
    cancelled: false,
  };
}

export function applyMenuKey(state, key = {}) {
  const { name, ctrl, sequence } = key;
  const total = state.items.length;

  if ((ctrl && name === 'c') || name === 'escape' || sequence === 'q' || sequence === 'Q') {
    return { ...state, cancelled: true, done: true };
  }
  if (name === 'up' || sequence === 'k' || sequence === 'K') {
    return { ...state, cursor: (state.cursor - 1 + total) % total, error: null };
  }
  if (name === 'down' || sequence === 'j' || sequence === 'J') {
    return { ...state, cursor: (state.cursor + 1) % total, error: null };
  }
  if (name === 'space' || sequence === ' ') {
    const nextSelected = new Set(state.selected);
    const id = state.items[state.cursor].id;
    if (nextSelected.has(id)) {
      nextSelected.delete(id);
    } else {
      nextSelected.add(id);
    }
    const check = validateSelection([...nextSelected], state.scope);
    return { ...state, selected: nextSelected, error: check.valid ? null : check.error };
  }
  if (sequence && /^[1-9]$/.test(sequence)) {
    const idx = Number(sequence) - 1;
    if (idx < total) {
      const nextSelected = new Set(state.selected);
      const id = state.items[idx].id;
      if (nextSelected.has(id)) {
        nextSelected.delete(id);
      } else {
        nextSelected.add(id);
      }
      const check = validateSelection([...nextSelected], state.scope);
      return { ...state, cursor: idx, selected: nextSelected, error: check.valid ? null : check.error };
    }
  }
  if (name === 'return' || name === 'enter') {
    const chosen = state.items.filter((h) => state.selected.has(h.id)).map((h) => h.id);
    const check = validateSelection(chosen, state.scope);
    if (!check.valid) {
      return { ...state, error: check.error };
    }
    return { ...state, done: true, error: null };
  }
  return state;
}

export function renderHostMenu(state, stdout = process.stdout) {
  const c = makeColors(colorsEnabled(stdout));
  const lines = [];
  lines.push(`${c.cyan('?')} ${c.bold(`Select coding agent host(s) to install (${state.scope}):`)}`);
  lines.push(c.dim('  (↑/↓ move, Space or 1-4 toggle, Enter confirm, q cancel)'));
  lines.push('');

  state.items.forEach((layout, idx) => {
    const focused = idx === state.cursor;
    const checked = state.selected.has(layout.id);
    const pointer = focused ? c.cyan('❯') : ' ';
    const box = checked ? c.green('◉') : c.dim('◯');
    const label = focused ? c.bold(layout.label.padEnd(15)) : layout.label.padEnd(15);
    const meta = c.dim(`(${layout.skillsRefPrefix} + ${layout.instructionDisplay})`);
    lines.push(`  ${pointer} ${box} ${label} ${meta}`);
  });

  const chosen = state.items.filter((h) => state.selected.has(h.id)).map((h) => h.id);
  const check = validateSelection(chosen, state.scope);
  lines.push('');
  if (state.error) {
    lines.push(`  ${c.red('✖')} ${c.red(state.error)}`);
  } else if (check.note) {
    lines.push(`  ${c.yellow('ℹ')} ${c.yellow(check.note)}`);
  } else {
    const names = state.items
      .filter((h) => state.selected.has(h.id))
      .map((h) => h.label)
      .join(', ');
    lines.push(`  ${c.dim(`Selected: ${names || 'none'}`)}`);
  }

  return lines;
}

async function runRawMenu({ stdin, stdout, initialState, applyKey, renderLines, onDone }) {
  let state = initialState;
  let renderedLines = 0;

  const draw = () => {
    if (renderedLines > 0) {
      stdout.write(`\x1b[${renderedLines}A\r\x1b[J`);
    }
    const lines = renderLines(state, stdout);
    renderedLines = lines.length;
    stdout.write(`${lines.join('\n')}\n`);
  };

  readline.emitKeypressEvents(stdin);
  const previousRaw = stdin.isRaw;
  stdin.setRawMode(true);
  stdin.resume();
  stdout.write('\x1b[?25l');

  try {
    draw();
    return await new Promise((resolve) => {
      const onKeypress = (_str, key) => {
        state = applyKey(state, key);
        if (state.done) {
          stdin.off('keypress', onKeypress);
          if (renderedLines > 0) {
            stdout.write(`\x1b[${renderedLines}A\r\x1b[J`);
          }
          resolve(onDone(state));
          return;
        }
        draw();
      };
      stdin.on('keypress', onKeypress);
    });
  } finally {
    stdout.write('\x1b[?25h');
    if (typeof stdin.setRawMode === 'function') {
      stdin.setRawMode(Boolean(previousRaw));
    }
    stdin.pause();
  }
}

export async function promptSelectScope({ stdin = process.stdin, stdout = process.stdout } = {}) {
  const c = makeColors(colorsEnabled(stdout));
  if (!stdin.isTTY || typeof stdin.setRawMode !== 'function') {
    stdout.write(`${c.cyan('?')} ${c.bold('Install scope:')}\n`);
    stdout.write('  1) Global (recommended, install once for all repos)\n');
    stdout.write('  2) Local  (install into a single project directory)\n');
    const rl = readline.createInterface({ input: stdin, output: stdout });
    try {
      const answer = await new Promise((resolve) => rl.question('Choose [1]: ', resolve));
      const trimmed = answer.trim().toLowerCase();
      if (trimmed === 'q') return null;
      return trimmed === '2' || trimmed === 'local' ? 'local' : 'global';
    } finally {
      rl.close();
    }
  }

  return runRawMenu({
    stdin,
    stdout,
    initialState: createScopeMenuState('global'),
    applyKey: applyScopeKey,
    renderLines: renderScopeMenu,
    onDone: (state) => {
      if (state.cancelled) {
        stdout.write(`${c.yellow('Cancelled.')}\n`);
        return null;
      }
      const chosen = state.items[state.cursor];
      stdout.write(`${c.green('✔')} ${c.bold('Scope:')} ${chosen.label}\n`);
      return chosen.id;
    },
  });
}

async function promptSelectHostsFallback(stdin, stdout, scope) {
  const c = makeColors(colorsEnabled(stdout));
  const items = CANONICAL_HOST_IDS.map((id) => getHostLayout(HOSTS[id], scope));
  stdout.write(`${c.cyan('?')} ${c.bold(`Select coding agent host(s) to install (${scope}):`)}\n`);
  items.forEach((h, i) => {
    stdout.write(`  ${i + 1}) ${h.label.padEnd(15)} (${h.skillsRefPrefix} + ${h.instructionDisplay})\n`);
  });

  const rl = readline.createInterface({ input: stdin, output: stdout });
  try {
    while (true) {
      const answer = await new Promise((resolve) => {
        rl.question('Enter number(s) separated by comma [1]: ', resolve);
      });
      const raw = answer.trim();
      if (raw.toLowerCase() === 'q') {
        return null;
      }
      const indices = (raw ? raw.split(',') : ['1']).map((t) => Number(t.trim()) - 1);
      if (indices.some((i) => !Number.isInteger(i) || i < 0 || i >= items.length)) {
        stdout.write(`${c.red(`Invalid selection. Enter numbers between 1 and ${items.length}.`)}\n`);
        continue;
      }
      const chosen = [...new Set(indices)].map((i) => items[i].id);
      const check = validateSelection(chosen, scope);
      if (!check.valid) {
        stdout.write(`${c.red(check.error)}\n`);
        continue;
      }
      return chosen;
    }
  } finally {
    rl.close();
  }
}

export async function promptSelectHosts({
  stdin = process.stdin,
  stdout = process.stdout,
  scope = 'global',
} = {}) {
  if (!stdin.isTTY || typeof stdin.setRawMode !== 'function') {
    return promptSelectHostsFallback(stdin, stdout, scope);
  }
  const c = makeColors(colorsEnabled(stdout));
  return runRawMenu({
    stdin,
    stdout,
    initialState: createHostMenuState(['codex'], scope),
    applyKey: applyMenuKey,
    renderLines: renderHostMenu,
    onDone: (state) => {
      if (state.cancelled) {
        stdout.write(`${c.yellow('Cancelled.')}\n`);
        return null;
      }
      const chosen = state.items.filter((h) => state.selected.has(h.id));
      const labels = chosen.map((h) => h.label).join(', ');
      stdout.write(`${c.green('✔')} ${c.bold('Hosts:')} ${labels}\n`);
      return chosen.map((h) => h.id);
    },
  });
}

export async function promptTargetDirectory({
  stdin = process.stdin,
  stdout = process.stdout,
  defaultDir = '.',
} = {}) {
  const c = makeColors(colorsEnabled(stdout));
  const resolvedPreview = path.resolve(defaultDir);
  const rl = readline.createInterface({ input: stdin, output: stdout });
  try {
    const answer = await new Promise((resolve) => {
      rl.question(
        `${c.cyan('?')} ${c.bold('Target project directory')} ${c.dim(`(current: ${resolvedPreview})`)}: `,
        resolve,
      );
    });
    const trimmed = answer.trim();
    return trimmed || defaultDir;
  } finally {
    rl.close();
  }
}

export async function runInteractiveInstallPrompt({
  pkg,
  stdin = process.stdin,
  stdout = process.stdout,
  presetScope = null,
  dirProvided = false,
  defaultDir = '.',
} = {}) {
  stdout.write(`\n${renderBanner(pkg, stdout)}\n\n`);
  const scope = presetScope || await promptSelectScope({ stdin, stdout });
  if (!scope) {
    return null;
  }
  const hostIds = await promptSelectHosts({ stdin, stdout, scope });
  if (!hostIds) {
    return null;
  }
  const targetDir = scope === 'local' && !dirProvided
    ? await promptTargetDirectory({ stdin, stdout, defaultDir })
    : defaultDir;
  return { scope, hostIds, targetDir };
}
