---
name: tilelang-review-skill
description: TileLang npuir 代码审查与格式校验技能。用户提及 review、代码审查、PR 前检查、lint、format、ruff、clang-format、规范检查、CI 不通过时必须使用本技能。优先识别行为回归、数值风险、同步风险与测试缺口，其次才是风格问题。
---

# TileLang Review Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Scope

- pre-PR code review for npuir branch
- format and lint checks aligned with CI
- risk-focused review for correctness, performance, and synchronization

## Review priorities

1. Behavior regressions
2. Precision and dtype risks
3. Synchronization and pipeline hazards
4. Missing tests
5. Style and format consistency

## Docs to consult first

- docs/Tilelang-Ascend贡献指南.md
- docs/Tilelang算子调试指南.md
- docs/开发指南.md

## Documentation drift check (agent/skill markdown changes)

When the change set touches `.opencode/agents/`, `.agents/skills/` (including
`_shared/standards/`), or the conductor orchestration docs, run the shared
standards drift checker before review conclusion:

```bash
python3 .agents/tools/standards_check.py check
```

- Exit 0: no drift. Exit 1: inspect the single-line JSON `failures[]`:
  - `SC-INLINE-DUP` — a consumer inlined canonical rule text instead of
    referencing `.agents/skills/_shared/standards/*.md` (reject: single
    source of truth violated);
  - `SC-HASH-MISMATCH` / `SC-UNTRACKED` — a standard file was edited
    without regenerating the lock (require the author to run
    `standards_check.py update`, and to confirm consumers of the changed
    standard were synchronized);
  - `SC-DANGLING-REF` / `SC-REF-PATH` / `SC-FP-STALE` — broken references
    (reject).
- Review focus for standards edits: the mechanical gate rules in
  `.agents/tools/gate_lint.py` must be updated in the same change when the
  edited standard carries a mechanically checkable subset.

## References

- references/checklist.txt

## Related skills

- tilelang-error-fixer
- tilelang-debug-helper
