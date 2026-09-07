"""Standards drift checker for _shared/standards (U10 fix, improvement #17).

Single source of truth governance for cross-agent shared rule blocks
(core-split strategy / algorithm research / negative-claim evidence /
gate & retry tables / stage-3 routing / final report template).

Checks (all mechanically decidable, no domain reasoning):
  - SC-LOCK-MISSING / SC-HASH-MISMATCH / SC-UNTRACKED
      standards .md files match standards.lock.json (SHA256 manifest);
      editing a standard without `standards_check.py update` is rejected.
  - SC-FP-STALE
      a drift fingerprint no longer exists in its canonical standard file
      (protects the detector itself from silent invalidation).
  - SC-INLINE-DUP
      a fingerprint appears verbatim in a consumer file (rule text was
      copy-pasted instead of referenced) — the core U10 drift signal.
  - SC-DANGLING-REF
      a consumer references _shared/standards/<file>.md that does not exist.
  - SC-REF-PATH
      repo-relative paths cited inside standards files must exist.

Usage:
  python3 standards_check.py check  [--repo-root PATH]   # default command
  python3 standards_check.py update [--repo-root PATH]   # regenerate lock
  python3 standards_check.py show   [--repo-root PATH]   # dump lock

Output: single-line JSON {ok, failures[], warnings[], stats}; exit 0 ok,
1 check failure, 2 usage error. Mirrors statectl conventions so the
conductor / review skill can consume it mechanically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

DEFAULT_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
STANDARDS_SUBDIR = os.path.join(".agents", "skills", "_shared", "standards")
LOCK_NAME = "standards.lock.json"

# Consumer scan scope (rule text must be referenced here, never inlined).
CONSUMER_SCANS = (
    (".opencode", "agents"),  # .opencode/agents/*.md
    (".agents", "skills"),  # .agents/skills/**/*.md (standards dir excluded)
)

# ---------------------------------------------------------------------------
# Drift fingerprints: verbatim sentinel substrings of canonical rule text.
# Each maps rule_id -> (canonical repo-relative file, sentinel string,
# allowed consumers). A sentinel must exist in its canonical file (else
# SC-FP-STALE) and must NOT appear in any consumer file outside the allowed
# set (else SC-INLINE-DUP).
#
# Canonical files may live inside _shared/standards/ (hash-locked) or be
# repo files that own a knowledge block by design (conductor stage table,
# scenario state machines, skill-evolution D/P/R/C table). The allowed set
# lists files where the sentinel may legitimately appear verbatim (the
# canonical owner itself + co-owners of scenario-specific variants).
# ---------------------------------------------------------------------------
_STD = ".agents/skills/_shared/standards"
FINGERPRINTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "SC-DUP-CORESPLIT": (
        f"{_STD}/core-split-strategy.md",
        "使其接近物理核数整数倍（按实查核数取 1×/2×/3×）",
        (f"{_STD}/core-split-strategy.md",),
    ),
    "SC-DUP-CORESPLIT2": (
        f"{_STD}/core-split-strategy.md",
        "禁止以文档假设或经验值（如 20~24）替代实查",
        (f"{_STD}/core-split-strategy.md",),
    ),
    "SC-DUP-RESEARCH": (
        f"{_STD}/algorithm-research.md",
        "复杂度对比须覆盖四口径（FLOPs / 访存 / 扫描遍数 / 中间缓冲峰值）",
        (f"{_STD}/algorithm-research.md",),
    ),
    "SC-DUP-EVIDENCE": (
        f"{_STD}/negative-claim-evidence.md",
        "禁止凭先验（尤其 GPU 直觉）在检索前否决候选",
        (f"{_STD}/negative-claim-evidence.md",),
    ),
    "SC-DUP-GATE": (
        f"{_STD}/gate-and-retry.md",
        "TileOPs 7 文件存在 + Tier 1 通过（import / manifest / collect-only）",
        (f"{_STD}/gate-and-retry.md",),
    ),
    "SC-DUP-STAGE3": (
        f"{_STD}/stage3-routing.md",
        "强制要求 Developer 先备份当前 impl 到 `history_version/{op}_impl_s3_attempt{N}.py` 再做修改",
        (f"{_STD}/stage3-routing.md",),
    ),
    "SC-DUP-PERFFEEDBACK": (
        f"{_STD}/perf-feedback.md",
        "参数级性能不足（tiling 可解）留在 Stage 4 迭代内，永不触发本机制",
        (f"{_STD}/perf-feedback.md",),
    ),
    "SC-DUP-REPORT": (
        f"{_STD}/final-report-template.md",
        "harness 集成验证: <smoke / full 用例数与结果，仅 migration-harness 填>",
        (f"{_STD}/final-report-template.md",),
    ),
    # --- C3.2 知识块指纹（变更半径治理）：状态机图 / Stage 表 / 目录结构 /
    # 重试上限表 / D-P-R-C 定义 / mode 枚举 / perf_records 契约 ---
    "SC-DUP-STATEMACHINE": (
        ".opencode/agents/conductor-scenarios/new-op.md",
        "INIT --> DESIGN --> REVIEW --> DEVELOP --> TUNING(可选) --> DONE",
        (
            ".opencode/agents/conductor-scenarios/new-op.md",
            ".opencode/agents/conductor-scenarios/migration.md",
            ".opencode/agents/conductor-scenarios/harness.md",
            ".opencode/agents/conductor-scenarios/optimize.md",
        ),
    ),
    "SC-DUP-STAGETABLE": (
        ".opencode/agents/tilelang-op-conductor.md",
        "| 4 算子调优 | `TUNING` | `@tilelang-op-optimizer` |",
        (".opencode/agents/tilelang-op-conductor.md",),
    ),
    "SC-DUP-DIRTREE": (
        ".opencode/agents/tilelang-op-conductor.md",
        "├── RETROSPECTIVE.md              # Stage 1/2/3/5 复盘（自进化钩子；Stage 4 复盘在 perf_opt/opt_log.md）",
        (".opencode/agents/tilelang-op-conductor.md",),
    ),
    "SC-DUP-RETRYLIMITS": (
        f"{_STD}/gate-and-retry.md",
        "5 次 Subagent 调度（运行失败 + 精度失败合并累计；`[DESIGN_ERROR]` 触发修订不计入）",
        (f"{_STD}/gate-and-retry.md",),
    ),
    "SC-DUP-DPRC": (
        ".agents/skills/tilelang-skill-evolution/SKILL.md",
        "| **D 实测数据** | 性能数字、代价常数、编译器/运行时陷阱实证、API 实际行为实证 | **Tier 0** 直接合入 `pattern-library.md` §1/§2（须带溯源 + 工具链版本戳 + 复现命令三件套） |",
        (
            ".agents/skills/tilelang-skill-evolution/SKILL.md",
            ".agents/skills/tilelang-skill-evolution/references/distillation-rules.md",
        ),
    ),
    "SC-DUP-MODEENUM": (
        f"{_STD}/signal-registry.md",
        "`first_impl` / `retry_impl` / `precision_fix`",
        (f"{_STD}/signal-registry.md",),
    ),
    "SC-DUP-OPTMODE": (
        f"{_STD}/signal-registry.md",
        "`full`（默认，完整调优流程） / `precision_fix`（仅精度回归修复）",
        (f"{_STD}/signal-registry.md",),
    ),
    "SC-DUP-PERFRECORDS": (
        f"{_STD}/signal-registry.md",
        "{round, candidate_id, parent_id, dispatch_path, workload, duration_us, l0_pass, msprof_raw_path, timestamp}",
        (f"{_STD}/signal-registry.md",),
    ),
}

# Repo-relative path tokens cited in prose (same shape as gate_lint.REF_PATH_RE).
REF_PATH_RE = re.compile(
    r"(?:docs|examples|testing|\.agents)/[^\s`'\"<>()\[\]（）【】：，。；、！？]+"
)
# References to shared standards files, e.g.
# `.agents/skills/_shared/standards/core-split-strategy.md` or relative
# `../../_shared/standards/core-split-strategy.md`.
STANDARDS_REF_RE = re.compile(r"_shared/standards/([A-Za-z0-9._\-]+\.md)")


def _fail(rule_id: str, file: str, message: str) -> dict:
    return {"rule_id": rule_id, "file": file, "message": message}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def standards_dir(repo_root: str) -> str:
    return os.path.join(repo_root, STANDARDS_SUBDIR)


def list_standard_files(repo_root: str) -> list[str]:
    sdir = standards_dir(repo_root)
    if not os.path.isdir(sdir):
        return []
    out = []
    for name in sorted(os.listdir(sdir)):
        if name.endswith(".md"):
            out.append(name)
    return out


def iter_consumer_files(repo_root: str) -> list[str]:
    """Markdown consumer files: .opencode/agents/*.md + .agents/skills/**/*.md,
    excluding the standards dir itself."""
    files: list[str] = []
    excluded_prefix = os.path.join(repo_root, STANDARDS_SUBDIR)
    for top in CONSUMER_SCANS:
        base = os.path.join(repo_root, *top)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(filenames):
                if not fn.endswith(".md"):
                    continue
                full = os.path.join(dirpath, fn)
                if os.path.commonpath([full, excluded_prefix]) == excluded_prefix:
                    continue
                files.append(os.path.relpath(full, repo_root))
    return files


def load_lock(repo_root: str) -> dict | None:
    lock_path = os.path.join(standards_dir(repo_root), LOCK_NAME)
    if not os.path.isfile(lock_path):
        return None
    data = _read_text(lock_path)
    if data is None:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"__corrupt__": True}


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def cmd_check(repo_root: str) -> tuple[dict, int]:
    failures: list[dict] = []
    warnings: list[dict] = []
    sdir = standards_dir(repo_root)
    sdir_rel = STANDARDS_SUBDIR

    if not os.path.isdir(sdir):
        print(
            json.dumps(
                {
                    "ok": False,
                    "failures": [
                        _fail(
                            "SC-NO-STANDARDS",
                            sdir_rel,
                            "标准目录不存在",
                        )
                    ],
                    "warnings": [],
                },
                ensure_ascii=False,
            )
        )
        return {}, 1

    std_files = list_standard_files(repo_root)
    lock = load_lock(repo_root)

    # --- lock presence / integrity ---
    if lock is None:
        failures.append(
            _fail(
                "SC-LOCK-MISSING",
                f"{sdir_rel}/{LOCK_NAME}",
                "lock 清单缺失：执行 `python3 .agents/tools/standards_check.py update` 生成",
            )
        )
        lock_files: dict[str, str] = {}
    elif lock.get("__corrupt__"):
        failures.append(
            _fail(
                "SC-LOCK-CORRUPT",
                f"{sdir_rel}/{LOCK_NAME}",
                "lock 清单 JSON 损坏，请重新 update",
            )
        )
        lock_files = {}
    else:
        lock_files = lock.get("files", {})
        if not lock_files:
            failures.append(
                _fail(
                    "SC-LOCK-EMPTY",
                    f"{sdir_rel}/{LOCK_NAME}",
                    "lock 清单为空，请重新 update",
                )
            )

    # --- hash match / untracked ---
    for name in std_files:
        actual = _sha256(os.path.join(sdir, name))
        if name not in lock_files:
            failures.append(
                _fail(
                    "SC-UNTRACKED",
                    f"{sdir_rel}/{name}",
                    "标准文件未登记进 lock：执行 update 后再提交",
                )
            )
        elif lock_files[name] != actual:
            failures.append(
                _fail(
                    "SC-HASH-MISMATCH",
                    f"{sdir_rel}/{name}",
                    "标准文件已修改但 lock 哈希未更新（执行 update 后再提交，"
                    "并同步受影响的消费者与 gate_lint 机械规则）",
                )
            )
    for name in lock_files:
        if name not in std_files and name != LOCK_NAME:
            failures.append(
                _fail(
                    "SC-LOCK-STALE",
                    f"{sdir_rel}/{name}",
                    "lock 登记的标准文件已被删除：执行 update 重新生成",
                )
            )

    # --- fingerprint health (canonical side; repo-relative canonical files) ---
    std_texts = {name: _read_text(os.path.join(sdir, name)) or "" for name in std_files}
    for rule_id, (canon, sentinel, _allowed) in FINGERPRINTS.items():
        canon_abs = os.path.join(repo_root, canon)
        canon_text = (
            std_texts[canon] if canon in std_texts else (_read_text(canon_abs) or "")
        )
        if not os.path.exists(canon_abs):
            failures.append(
                _fail(
                    "SC-FP-STALE",
                    canon,
                    f"指纹 {rule_id} 的权威文件不存在——同步更新 FINGERPRINTS 表",
                )
            )
        elif sentinel not in canon_text:
            failures.append(
                _fail(
                    "SC-FP-STALE",
                    canon,
                    f"指纹 {rule_id} 已不存在于权威文件（标准文本变更后须同步 FINGERPRINTS 表）",
                )
            )

    # --- consumer scan: inline duplication + dangling standards refs ---
    consumers = iter_consumer_files(repo_root)
    dup_hits = 0
    ref_hits = 0
    for rel in consumers:
        text = _read_text(os.path.join(repo_root, rel))
        if text is None:
            continue
        for rule_id, (_canon, sentinel, allowed) in FINGERPRINTS.items():
            if sentinel in text and rel not in allowed:
                dup_hits += 1
                failures.append(
                    _fail(
                        "SC-INLINE-DUP",
                        rel,
                        f"内联了标准规则文本（指纹 {rule_id}，权威出处 {_canon}）："
                        f"改为引用权威文件路径，删除整段复制",
                    )
                )
        for m in STANDARDS_REF_RE.finditer(text):
            target = m.group(1)
            if target == LOCK_NAME:
                continue
            ref_hits += 1
            if target not in std_files:
                failures.append(
                    _fail("SC-DANGLING-REF", rel, f"引用的标准文件不存在：{m.group(0)}")
                )

    # --- repo paths cited inside standards files must exist ---
    for name, text in std_texts.items():
        for m in REF_PATH_RE.finditer(text):
            token = m.group(0).rstrip(".,;:!?、。】")
            if "{" in token or "}" in token or "*" in token:
                continue  # template variables / globs
            if not os.path.exists(os.path.join(repo_root, token)):
                # *.md wildcard-style references to a directory are allowed
                if token.endswith(".md") and os.path.isdir(
                    os.path.dirname(os.path.join(repo_root, token))
                ):
                    continue
                failures.append(
                    _fail(
                        "SC-REF-PATH",
                        f"{sdir_rel}/{name}",
                        f"标准文件引用的仓库路径不存在：{token}",
                    )
                )

    result = {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "stats": {
            "standards_files": len(std_files),
            "consumers_scanned": len(consumers),
            "standards_refs": ref_hits,
            "dup_hits": dup_hits,
        },
    }
    print(json.dumps(result, ensure_ascii=False))
    return result, (0 if not failures else 1)


# ---------------------------------------------------------------------------
# update / show
# ---------------------------------------------------------------------------


def cmd_update(repo_root: str) -> int:
    sdir = standards_dir(repo_root)
    if not os.path.isdir(sdir):
        print(
            json.dumps(
                {"ok": False, "error": f"标准目录不存在：{STANDARDS_SUBDIR}"},
                ensure_ascii=False,
            )
        )
        return 1
    files = {
        name: _sha256(os.path.join(sdir, name))
        for name in list_standard_files(repo_root)
    }
    lock = {"files": dict(sorted(files.items()))}
    lock_path = os.path.join(sdir, LOCK_NAME)
    with open(lock_path, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        json.dumps(
            {
                "ok": True,
                "updated": sorted(files),
                "lock": os.path.relpath(lock_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_show(repo_root: str) -> int:
    lock = load_lock(repo_root)
    if lock is None:
        print(json.dumps({"ok": False, "error": "lock 清单缺失"}, ensure_ascii=False))
        return 1
    print(json.dumps(lock, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="standards_check", description="共享标准单一事实源漂移检查（U10）"
    )
    parser.add_argument(
        "command", nargs="?", default="check", choices=["check", "update", "show"]
    )
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    args = parser.parse_args(argv)
    repo_root = os.path.abspath(args.repo_root)
    if args.command == "update":
        return cmd_update(repo_root)
    if args.command == "show":
        return cmd_show(repo_root)
    _, code = cmd_check(repo_root)
    return code


if __name__ == "__main__":
    sys.exit(main())
