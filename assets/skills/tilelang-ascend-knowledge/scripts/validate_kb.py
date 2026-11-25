#!/usr/bin/env python3
"""Validate structure, metadata, index coverage, links, and hygiene of the Ascend TileLang KB."""

import argparse
import json
from pathlib import Path
import re
import sys

CARD_DIRS = ("foundations", "patterns", "families")
VALID_LAYERS = {"foundation", "pattern", "falsified", "family"}
VALID_KINDS = {"methodology", "hw-pattern", "compiler-note", "falsified-direction", "playbook"}
VALID_STATUSES = {"active", "draft", "superseded"}
REQUIRED_FIELDS = ("id", "layer", "arch", "kind", "status", "verified_scope")

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
PRIVATE_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|/root/[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\)"
)


def parse_frontmatter(text):
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return None
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return None
    data = {}
    for line in lines[1:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            return None
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            data[key] = items
        else:
            data[key] = value.strip("'\"")
    return data


def iter_markdown_links(text):
    in_code_block = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        for match in LINK_RE.finditer(line):
            yield match.group(1).strip()


def expected_layer(rel_path):
    parts = rel_path.parts
    if parts[0] == "foundations":
        return "foundation"
    if parts[0] == "patterns":
        return "falsified" if len(parts) > 1 and parts[1] == "falsified" else "pattern"
    if parts[0] == "families":
        return "family"
    return None


def validate_kb(kb_root):
    kb_root = Path(kb_root).resolve()
    errors = []
    warnings = []

    skill_md = kb_root / "SKILL.md"
    index_md = kb_root / "INDEX.md"
    if not skill_md.is_file():
        errors.append("missing SKILL.md")
    if not index_md.is_file():
        errors.append("missing INDEX.md")
        return {"ok": False, "errors": errors, "warnings": warnings, "cards": 0}

    index_text = index_md.read_text(encoding="utf-8")
    indexed_targets = set()
    for target in iter_markdown_links(index_text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        resolved = (index_md.parent / clean).resolve()
        if not resolved.exists():
            errors.append(f"INDEX.md broken link: {target}")
        elif resolved.is_relative_to(kb_root):
            indexed_targets.add(resolved.relative_to(kb_root))

    cards = []
    seen_ids = {}
    for top in CARD_DIRS:
        base = kb_root / top
        if not base.is_dir():
            errors.append(f"missing required layer directory: {top}/")
            continue
        for md in sorted(base.rglob("*.md")):
            rel = md.relative_to(kb_root)
            cards.append(rel)
            text = md.read_text(encoding="utf-8")

            if PRIVATE_PATH_RE.search(text):
                errors.append(f"{rel}: contains private host path")

            meta = parse_frontmatter(text)
            if meta is None:
                errors.append(f"{rel}: missing or malformed YAML frontmatter")
            else:
                for field in REQUIRED_FIELDS:
                    val = meta.get(field)
                    if not val:
                        errors.append(f"{rel}: missing required frontmatter field '{field}'")
                card_id = meta.get("id")
                if card_id:
                    if card_id in seen_ids:
                        errors.append(f"{rel}: duplicate card id '{card_id}' (also in {seen_ids[card_id]})")
                    seen_ids[card_id] = str(rel)
                layer = meta.get("layer")
                exp_layer = expected_layer(rel)
                if layer not in VALID_LAYERS:
                    errors.append(f"{rel}: invalid layer '{layer}'")
                elif exp_layer and layer != exp_layer:
                    errors.append(f"{rel}: layer '{layer}' does not match directory (expected '{exp_layer}')")
                if meta.get("kind") not in VALID_KINDS:
                    errors.append(f"{rel}: invalid kind '{meta.get('kind')}'")
                if meta.get("status") not in VALID_STATUSES:
                    errors.append(f"{rel}: invalid status '{meta.get('status')}'")
                arch = meta.get("arch")
                if isinstance(arch, list) and not arch:
                    errors.append(f"{rel}: arch list must not be empty")

            if rel not in indexed_targets:
                errors.append(f"{rel}: orphan card not linked in INDEX.md")

            for target in iter_markdown_links(text):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                clean = target.split("#", 1)[0]
                if not clean:
                    continue
                resolved = (md.parent / clean).resolve()
                if not resolved.exists():
                    errors.append(f"{rel}: broken relative link '{target}'")

    # Check SKILL.md, INDEX.md, and templates for private path leaks and broken links
    for extra in [skill_md, index_md, *sorted((kb_root / "templates").glob("*.md"))]:
        if not extra.is_file():
            continue
        rel = extra.relative_to(kb_root)
        text = extra.read_text(encoding="utf-8")
        if PRIVATE_PATH_RE.search(text):
            errors.append(f"{rel}: contains private host path")
        if extra != index_md:
            for target in iter_markdown_links(text):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                clean = target.split("#", 1)[0]
                if not clean:
                    continue
                resolved = (extra.parent / clean).resolve()
                if not resolved.exists():
                    errors.append(f"{rel}: broken relative link '{target}'")

    return {
        "ok": not errors,
        "cards": len(cards),
        "indexed": len(indexed_targets),
        "errors": errors,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to the tilelang-ascend-knowledge root directory",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()
    result = validate_kb(args.root)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        status = "OK" if result["ok"] else "FAILED"
        print(f"[{status}] cards={result['cards']} indexed={result['indexed']} errors={len(result['errors'])}")
        for err in result["errors"]:
            print(f"  ERROR: {err}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
