#!/usr/bin/env python3
"""Install shared Harness skills into an explicitly selected workspace."""

import argparse
from pathlib import Path
import shutil
import sys


SOURCE = Path(__file__).resolve().parents[1] / "assets" / "skills"
NAMES = (
    "harness-new-project", "harness-onboard", "harness-resume",
    "harness-prepare-environment", "harness-establish-baseline", "harness-port-baseline", "harness-analysis",
    "harness-develop",
    "harness-status", "harness-next", "harness-automatic",
    "harness-archive-project", "harness-restore-project",
)


def contents(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def install(project):
    root = Path(project).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project-dir must be an existing directory")
    target = root / ".agents" / "skills"
    pending = []
    for name in NAMES:
        destination = target / name
        if destination.exists() or destination.is_symlink():
            if not destination.is_dir() or any(
                not (destination / path).is_file() or (destination / path).read_bytes() != data
                for path, data in contents(SOURCE / name).items()
            ):
                raise ValueError(f"existing skill differs; files preserved: {destination}")
        else:
            pending.append(name)
    instructions = root / "AGENTS.md"
    existing = instructions.read_text() if instructions.exists() else ""
    entry = (SOURCE.parent / "codex" / "AGENTS.md").read_text()
    start, end = "<!-- operator-harness -->", "<!-- /operator-harness -->"
    if start in existing or end in existing:
        if existing.count(start) != 1 or existing.count(end) != 1:
            raise ValueError("ambiguous Harness AGENTS block; files preserved")
        block = existing[existing.index(start):existing.index(end) + len(end)]
        if block != entry.strip():
            raise ValueError("existing Harness AGENTS block differs; review its upgrade before installing; files preserved")
    for name in pending:
        shutil.copytree(SOURCE / name, target / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    ignore = root / ".gitignore"
    existing_ignore = ignore.read_text() if ignore.exists() else ""
    if ".workspace/" not in existing_ignore.splitlines():
        with ignore.open("a") as output:
            output.write(("\n" if existing_ignore else "") + ".workspace/\n")
    if "<!-- operator-harness -->" not in existing:
        with instructions.open("a") as output:
            output.write(("\n\n" if existing else "") + entry)
    return target, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", "--project-dir", dest="project_dir", required=True,
                        help="exact workspace root; --project-dir is a compatibility alias")
    args = parser.parse_args()
    try:
        target, installed = install(args.project_dir)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Installed {', '.join(installed)} in {target}" if installed
          else f"Skills already installed in {target}")
    print(f"Project entry: {target.parent.parent / 'AGENTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
