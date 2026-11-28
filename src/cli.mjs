import os from 'node:os';
import {
  adoptLegacy,
  applyUpgrade,
  doctorWorkspace,
  getPackageInfo,
  installWorkspace,
  prepareUpgrade,
  resolveDirectory,
  runTransaction,
  uninstallWorkspace,
} from './bundle.mjs';
import {
  CANONICAL_HOST_IDS,
  HOSTS,
  discoverInstallations,
  getHostLayout,
  parseHostList,
  resolveHost,
} from './hosts.mjs';
import { renderBanner, runInteractiveInstallPrompt } from './prompt.mjs';

function formatHelp(pkg) {
  return `${pkg.name} v${pkg.version} — Operator Harness CLI

Usage:
  tilelang-harness                  Launch interactive installer (in TTY)
  tilelang-harness <command> [options]

Commands:
  init        Install Harness skills and instructions (global or project-local)
  doctor      Check installation integrity and local runtime
  upgrade     Plan or apply a managed transactional upgrade
  resume      Resume an interrupted upgrade or rollback transaction
  rollback    Roll back a completed or interrupted upgrade transaction
  adopt       Record manifest ownership for an exact legacy installation
  uninstall   Remove managed Harness files (preserves user files & .workspace/)
  hosts       List supported coding agent hosts and global/local paths

Common Options:
  -g, --global            Install/manage globally in user home (~/.codex, ~/.claude, ...)
  -l, --local             Install/manage locally in a project directory
  -d, --dir <path>        Target project directory for --local (default: ".")
                          Aliases: -w, --workspace, --project-dir
      --host <name>       Target host: codex | antigravity | claude-code | agent
                          Aliases: agy, claude, .agent, generic
                          (init accepts comma-separated or repeated --host)
  -y, --yes               Skip interactive prompts and use defaults
      --json              Output machine-readable JSON
  -h, --help              Show help
  -v, --version           Show version

Examples:
  npx github:tile-ai/tilelang-mlir-ascend#harness
  npx github:tile-ai/tilelang-mlir-ascend#harness init --global --host codex,claude-code
  npx github:tile-ai/tilelang-mlir-ascend#harness init --local --dir /path/to/repo --host codex
  npx github:tile-ai/tilelang-mlir-ascend#harness doctor
  npx github:tile-ai/tilelang-mlir-ascend#harness upgrade --plan --global --host codex
  npx github:tile-ai/tilelang-mlir-ascend#harness upgrade --apply <PLAN_ID> --global --host codex
  npx github:tile-ai/tilelang-mlir-ascend#harness rollback <TRANSACTION_ID> --global --host codex
  npx github:tile-ai/tilelang-mlir-ascend#harness uninstall --global --host codex --dry-run
`;
}

function parseArgs(argv) {
  const positionals = [];
  const flags = {
    dir: '.',
    dirProvided: false,
    global: false,
    local: false,
    hosts: [],
    check: false,
    plan: false,
    apply: null,
    from: null,
    dryRun: false,
    force: false,
    yes: false,
    json: false,
    help: false,
    version: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) {
        throw new Error(`missing value for ${arg}`);
      }
      i += 1;
      return argv[i];
    };

    if (arg === '-h' || arg === '--help') {
      flags.help = true;
    } else if (arg === '-v' || arg === '--version') {
      flags.version = true;
    } else if (arg === '-g' || arg === '--global') {
      flags.global = true;
    } else if (arg === '-l' || arg === '--local') {
      flags.local = true;
    } else if (arg === '-y' || arg === '--yes' || arg === '--non-interactive') {
      flags.yes = true;
    } else if (arg === '-d' || arg === '--dir' || arg === '-w' || arg === '--workspace' || arg === '--project-dir') {
      flags.dir = next();
      flags.dirProvided = true;
    } else if (arg.startsWith('--dir=')) {
      flags.dir = arg.slice('--dir='.length);
      flags.dirProvided = true;
    } else if (arg.startsWith('--workspace=')) {
      flags.dir = arg.slice('--workspace='.length);
      flags.dirProvided = true;
    } else if (arg.startsWith('--project-dir=')) {
      flags.dir = arg.slice('--project-dir='.length);
      flags.dirProvided = true;
    } else if (arg === '--host') {
      flags.hosts.push(next());
    } else if (arg.startsWith('--host=')) {
      flags.hosts.push(arg.slice('--host='.length));
    } else if (arg === '--check') {
      flags.check = true;
    } else if (arg === '--plan') {
      flags.plan = true;
    } else if (arg === '--apply') {
      flags.apply = next();
    } else if (arg.startsWith('--apply=')) {
      flags.apply = arg.slice('--apply='.length);
    } else if (arg === '--from' || arg === '--adopt-from') {
      flags.from = next();
    } else if (arg.startsWith('--from=')) {
      flags.from = arg.slice('--from='.length);
    } else if (arg === '--dry-run') {
      flags.dryRun = true;
    } else if (arg === '--force') {
      flags.force = true;
    } else if (arg === '--json') {
      flags.json = true;
    } else if (arg.startsWith('-')) {
      throw new Error(`unknown option: ${arg}`);
    } else {
      positionals.push(arg);
    }
  }

  if (flags.global && flags.local) {
    throw new Error('cannot combine --global and --local');
  }
  if (flags.global && flags.dirProvided) {
    throw new Error('cannot combine --global with --dir/--workspace');
  }

  return { command: positionals[0] || null, args: positionals.slice(1), flags };
}

function explicitScope(flags) {
  if (flags.global) return 'global';
  if (flags.local || flags.dirProvided) return 'local';
  return null;
}

function resolveSingleTarget(flags, homeDir, { allowAutoDiscover = true, defaultScope = 'local' } = {}) {
  const scopeChoice = explicitScope(flags);
  if (scopeChoice) {
    const baseRoot = scopeChoice === 'global'
      ? resolveDirectory(homeDir, 'home directory')
      : resolveDirectory(flags.dir, 'target directory');
    if (flags.hosts.length > 0) {
      const list = parseHostList(flags.hosts, scopeChoice);
      if (list.length !== 1) {
        throw new Error('this command expects a single --host value');
      }
      return { scope: scopeChoice, layout: list[0] };
    }
    if (allowAutoDiscover) {
      const discovered = discoverInstallations(baseRoot, scopeChoice);
      if (discovered.length === 1) {
        return { scope: scopeChoice, layout: discovered[0].layout };
      }
      if (discovered.length > 1) {
        const names = discovered.map((d) => d.layout.id).join(', ');
        throw new Error(`multiple ${scopeChoice} installations detected (${names}); specify --host <name>`);
      }
    }
    return { scope: scopeChoice, layout: getHostLayout(HOSTS.codex, scopeChoice) };
  }

  if (allowAutoDiscover) {
    const localRoot = resolveDirectory(flags.dir, 'target directory');
    const globalRoot = resolveDirectory(homeDir, 'home directory');
    const localFound = discoverInstallations(localRoot, 'local');
    const globalFound = discoverInstallations(globalRoot, 'global');
    const allFound = [...localFound, ...globalFound];

    if (flags.hosts.length > 0) {
      const requestedId = resolveHost(flags.hosts[0]).id;
      const matches = allFound.filter((item) => item.layout.id === requestedId);
      if (matches.length === 1) {
        return { scope: matches[0].scope, layout: matches[0].layout };
      }
      if (matches.length > 1) {
        throw new Error(`host "${requestedId}" is installed in both local and global scopes; specify --local or --global`);
      }
      return { scope: defaultScope, layout: getHostLayout(requestedId, defaultScope) };
    }

    if (allFound.length === 1) {
      return { scope: allFound[0].scope, layout: allFound[0].layout };
    }
    if (allFound.length > 1) {
      const summary = allFound.map((d) => `${d.scope}:${d.layout.id}`).join(', ');
      throw new Error(`multiple installations detected (${summary}); specify --global/--local and --host <name>`);
    }
  }

  const hostId = flags.hosts.length > 0 ? resolveHost(flags.hosts[0]).id : 'codex';
  return { scope: defaultScope, layout: getHostLayout(hostId, defaultScope) };
}

function printJson(value, stdout) {
  stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function formatDoctorHuman(report) {
  const lines = [];
  if (report.workspace) {
    lines.push(`Local Dir: ${report.workspace}`);
  }
  if (report.home_dir) {
    lines.push(`Home Dir:  ${report.home_dir}`);
  }
  lines.push(`Package:   ${report.package.name}@${report.package.version}`);
  lines.push(`Runtime:   node ${report.runtime.node} (${report.runtime.platform})`);
  if (report.runtime.python.available) {
    lines.push(`Python:    ${report.runtime.python.command} (${report.runtime.python.version})`);
  } else {
    lines.push('Python:    not found');
  }

  for (const inst of report.installations) {
    const badge = inst.ok ? 'OK' : 'FAIL';
    lines.push(
      `Host [${inst.scope}:${inst.host}]: ${badge} (mode=${inst.mode}, skills=${inst.skills_dir}, entry=${inst.instruction_file})`,
    );
    for (const issue of inst.issues) {
      lines.push(`  [error] ${issue}`);
    }
    for (const warn of inst.warnings) {
      lines.push(`  [warn]  ${warn}`);
    }
  }
  for (const issue of report.issues) {
    lines.push(`[error] ${issue}`);
  }
  for (const warn of report.warnings) {
    lines.push(`[warn]  ${warn}`);
  }
  lines.push(`Overall: ${report.ok ? 'HEALTHY' : 'ISSUES FOUND'}`);
  return `${lines.join('\n')}\n`;
}

function isInteractiveTty(io) {
  return Boolean(io.stdin && io.stdin.isTTY && io.stdout && io.stdout.isTTY);
}

export async function main(
  argv,
  io = { stdin: process.stdin, stdout: process.stdout, stderr: process.stderr, homeDir: os.homedir() },
) {
  const streams = {
    stdin: io.stdin ?? process.stdin,
    stdout: io.stdout ?? process.stdout,
    stderr: io.stderr ?? process.stderr,
    homeDir: io.homeDir ?? os.homedir(),
  };
  const pkg = getPackageInfo();
  try {
    const { command: rawCommand, args, flags } = parseArgs(argv);

    if (flags.version && !rawCommand) {
      streams.stdout.write(`${pkg.version}\n`);
      return 0;
    }
    if (flags.help) {
      streams.stdout.write(formatHelp(pkg));
      return 0;
    }

    const interactive = isInteractiveTty(streams);
    let command = rawCommand;
    if (!command) {
      if (interactive && !flags.json && !flags.yes) {
        command = 'init';
      } else {
        streams.stdout.write(formatHelp(pkg));
        return 0;
      }
    }

    if (command === 'hosts') {
      const list = CANONICAL_HOST_IDS.map((id) => ({
        id,
        label: HOSTS[id].label,
        aliases: HOSTS[id].aliases,
        global: getHostLayout(HOSTS[id], 'global'),
        local: getHostLayout(HOSTS[id], 'local'),
      }));
      if (flags.json) {
        printJson(list, streams.stdout);
      } else {
        for (const h of list) {
          const aliasText = h.aliases.length ? ` (aliases: ${h.aliases.join(', ')})` : '';
          streams.stdout.write(`${h.id} — ${h.label}${aliasText}\n`);
          streams.stdout.write(`  global: ${h.global.skillsRefPrefix.padEnd(25)} + ${h.global.instructionDisplay}\n`);
          streams.stdout.write(`  local:  ${h.local.skillsRefPrefix.padEnd(25)} + ${h.local.instructionDisplay}\n`);
        }
      }
      return 0;
    }

    if (command === 'init') {
      let hostFlags = flags.hosts;
      let targetDir = flags.dir;
      let scope = explicitScope(flags);
      const shouldPrompt = interactive
        && hostFlags.length === 0
        && !flags.yes
        && !flags.json
        && !flags.check;

      if (shouldPrompt) {
        const prompted = await runInteractiveInstallPrompt({
          pkg,
          stdin: streams.stdin,
          stdout: streams.stdout,
          presetScope: scope,
          dirProvided: flags.dirProvided,
          defaultDir: targetDir,
        });
        if (!prompted) {
          return 130;
        }
        scope = prompted.scope;
        hostFlags = prompted.hostIds;
        targetDir = prompted.targetDir;
      } else {
        // Default non-interactive scope: --global unless --local / --dir / --workspace was specified.
        scope = scope || 'global';
        if (!flags.json && !flags.check && interactive) {
          streams.stdout.write(`\n${renderBanner(pkg, streams.stdout)}\n\n`);
        }
      }

      const layouts = parseHostList(hostFlags, scope);
      // Pre-validate every requested host before writing any host.
      for (const layout of layouts) {
        installWorkspace({
          workspace: targetDir,
          scope,
          homeDir: streams.homeDir,
          host: layout,
          checkOnly: true,
        });
      }
      const results = layouts.map((layout) =>
        installWorkspace({
          workspace: targetDir,
          scope,
          homeDir: streams.homeDir,
          host: layout,
          checkOnly: flags.check,
        }),
      );
      if (flags.json) {
        printJson(results.length === 1 ? results[0] : results, streams.stdout);
      } else {
        for (const res of results) {
          if (flags.check) {
            streams.stdout.write(
              `[check] scope=${res.scope} host=${res.host} status=${res.status} skills_dir=${res.skills_dir}\n`,
            );
          } else if (res.installed_skills.length > 0) {
            streams.stdout.write(
              `✔ Installed ${res.installed_skills.length} skills (${res.scope}) for ${res.host} → ${res.skills_dir}\n`,
            );
          } else {
            streams.stdout.write(
              `✔ Skills already installed (${res.scope}) for ${res.host} → ${res.skills_dir}\n`,
            );
          }
          streams.stdout.write(`  Entry:    ${res.instruction_file}\n`);
          if (res.manifest_path) {
            streams.stdout.write(`  Manifest: ${res.manifest_path}\n`);
          }
        }
        if (!flags.check) {
          streams.stdout.write('\nNext steps:\n');
          streams.stdout.write('  • Open any operator repository and start with $harness-new-project or $harness-onboard\n');
          streams.stdout.write('  • Each repository uses its own .workspace/ at runtime; to keep shared .gitignore clean, run:\n');
          streams.stdout.write('      echo ".workspace/" >> "$(git rev-parse --path-format=absolute --git-path info/exclude)"\n');
          streams.stdout.write('  • Run "tilelang-harness doctor" to verify your setup\n');
        }
      }
      return 0;
    }

    if (command === 'doctor') {
      const scope = explicitScope(flags) || 'auto';
      const host = flags.hosts.length > 0 ? resolveHost(flags.hosts[0]) : null;
      const report = doctorWorkspace({
        workspace: flags.dir,
        scope,
        homeDir: streams.homeDir,
        host,
      });
      if (flags.json) {
        printJson(report, streams.stdout);
      } else {
        streams.stdout.write(formatDoctorHuman(report));
      }
      return report.ok ? 0 : 1;
    }

    if (command === 'upgrade') {
      const target = resolveSingleTarget(flags, streams.homeDir);
      const planId = flags.apply || args[0] || null;
      if (flags.plan && planId) {
        throw new Error('use either --plan or --apply <PLAN_ID>, not both');
      }
      if (flags.plan || !planId) {
        if (!flags.plan && !planId) {
          throw new Error('specify --plan to preview or --apply <PLAN_ID> to apply an upgrade');
        }
        const { plan } = prepareUpgrade({
          workspace: flags.dir,
          scope: target.scope,
          homeDir: streams.homeDir,
          host: target.layout,
        });
        printJson(plan, streams.stdout);
        return 0;
      }
      const result = applyUpgrade({
        workspace: flags.dir,
        scope: target.scope,
        homeDir: streams.homeDir,
        host: target.layout,
        planId,
      });
      printJson(result, streams.stdout);
      return 0;
    }

    if (command === 'resume' || command === 'rollback') {
      const transactionId = args[0];
      if (!transactionId) {
        throw new Error(`missing <TRANSACTION_ID> for ${command}`);
      }
      const target = resolveSingleTarget(flags, streams.homeDir);
      const result = runTransaction({
        workspace: flags.dir,
        scope: target.scope,
        homeDir: streams.homeDir,
        host: target.layout,
        transactionId,
        rollback: command === 'rollback',
      });
      printJson(result, streams.stdout);
      return 0;
    }

    if (command === 'adopt') {
      if (!flags.from) {
        throw new Error('adopt requires --from <BUNDLE_ROOT>');
      }
      const target = resolveSingleTarget(flags, streams.homeDir, {
        allowAutoDiscover: false,
        defaultScope: 'local',
      });
      const result = adoptLegacy({
        workspace: flags.dir,
        scope: target.scope,
        homeDir: streams.homeDir,
        host: target.layout,
        bundleRoot: flags.from,
      });
      printJson(result, streams.stdout);
      return 0;
    }

    if (command === 'uninstall') {
      const target = resolveSingleTarget(flags, streams.homeDir);
      const result = uninstallWorkspace({
        workspace: flags.dir,
        scope: target.scope,
        homeDir: streams.homeDir,
        host: target.layout,
        dryRun: flags.dryRun,
        force: flags.force,
      });
      if (flags.json) {
        printJson(result, streams.stdout);
      } else if (flags.dryRun) {
        streams.stdout.write(
          `[dry-run] Would remove ${result.files_to_remove.length} managed files (${result.scope}) for ${result.host} (${result.instruction_file}: ${result.instruction_action}); .workspace/ preserved.\n`,
        );
      } else {
        streams.stdout.write(
          `Uninstalled ${result.removed_files} managed files (${result.scope}) for ${result.host} (${result.instruction_file}: ${result.instruction_action}); .workspace/ preserved.\n`,
        );
      }
      return 0;
    }

    throw new Error(`unknown command "${command}"; run "tilelang-harness --help"`);
  } catch (err) {
    streams.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
    return 1;
  }
}
