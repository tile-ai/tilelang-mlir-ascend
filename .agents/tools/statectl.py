"""statectl — deterministic state-machine CLI for tilelang-op-conductor.

Fixes U1（状态机全靠 LLM 手工读写，无机械校验）from
docs/developer/conductor-improvement-analysis-aggregated.md（改进项 #1/#2）;
U14 修复（改进项 #20）追加 [DESIGN_LIMIT] 逆向反馈路由记录
（``set --perf-feedback-action``）与 perf_feedback.md 的 gate 4 schema 校验;
改进项 #1.3（任务时间线事件流）补全事件字段
（stage/subagent/mode/verdict/duration_s）与 ``timeline-summary``
机械汇总（最终报告「时间线」段 + evolver 失败根因链一手输入）;
A2（改进分析 2026-09）追加 ``set --perf-iteration-*`` 三个白名单 flag
（修复 perf_iteration 字段与写入接口的断线）与 gate 4 对
perf_opt/perf_records.jsonl 的记录校验 + winner 对账;
E1.3 追加 ``budget`` 字段（``set --budget-json``）与 ``start`` 的
预算水位回报（advisory，超限经 budget.exceeded 交给 conductor 路由）。

The conductor calls this tool via bash instead of hand-editing
``.stage_state.json`` / ``.migration_state.json``. Mechanical duties only
(zero domain reasoning — 分核数值/算法结论/等价性推演仍归 LLM):

  - transition legality: skip-stage / duplicate complete / concurrent
    in_progress / retry-budget interception
  - counter migration: retry_count & stage_retry_count increments, limits,
    BLOCKED_* terminal codes
  - structured gates (``gate`` / auto-gate inside ``complete``): JSON output
    with rule_id + message, exit code 1 on failure and no state write
  - artifact SHA256 snapshot (auto after complete) + drift detection (``verify``)
  - ``.task_timeline.jsonl`` event stream for every mutation (1.3: each line
    carries ts / action / stage / subagent / mode / verdict / duration_s)
    plus read-only ``timeline-summary`` (per-stage attempts & durations,
    dispatch count, chronological failure chain)
  - schema validation + missing-field backfill (``repair --apply``)

Exit codes: 0 = ok; 1 = validation/gate failure (see JSON ``errors``);
2 = usage error (argparse).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_lint  # noqa: E402

STATE_FILE = ".stage_state.json"
MIGRATION_STATE_FILE = ".migration_state.json"
TIMELINE_FILE = ".task_timeline.jsonl"

STAGE_PHASE = {
    0: "SCAFFOLD",
    1: "DESIGN",
    2: "REVIEW",
    3: "DEVELOP",
    4: "TUNING",
    5: "INTEGRATE",
}
TERMINAL_PHASES = {"DONE", "FAILED"}
STAGE_RETRY_LIMITS = {0: 3, 1: 3, 3: 5, 4: 10, 5: 2}
BLOCKED_BY_STAGE = {
    0: "BLOCKED_SCAFFOLD",
    1: "BLOCKED_DESIGN",
    3: "BLOCKED_IMPL",
    5: "BLOCKED_INTEGRATION",
}
MIG_PHASES = ["SCAFFOLD", "DEV_LOOP", "INTEGRATE", "DONE", "FAILED"]
FUNC_STATUSES = {"pending", "in_progress", "done", "failed"}

# Stage → subagent dispatch mapping — mechanical mirror of the conductor
# 「核心调度流程」table; used ONLY for timeline observability labels
# (zero domain reasoning, no scheduling logic lives here).
STAGE_SUBAGENT = {
    0: "tileops-scaffolder",
    1: "tilelang-op-designer",
    2: "tilelang-design-reviewer",
    3: "tilelang-op-developer",
    4: "tilelang-op-optimizer",
    5: "tilelang-op-integrator",
}
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def now_iso() -> str:
    fake = os.environ.get("STATECTL_NOW")  # determinism hook for tests
    if fake:
        return fake
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def err(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def emit(payload: dict, exit_code: int) -> int:
    payload.setdefault("ok", exit_code == 0)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return exit_code


def find_repo_root(start: str) -> str:
    cur = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(cur, ".git")) or os.path.exists(
            os.path.join(cur, "AGENTS.md")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


def atomic_write_json(path: str, data: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".statectl-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def rel_posix(repo_root: str, path: str) -> str:
    return os.path.relpath(os.path.abspath(path), repo_root).replace(os.sep, "/")


def append_timeline(op_dir: str, event: dict) -> None:
    event.setdefault("ts", now_iso())
    path = os.path.join(op_dir, TIMELINE_FILE)
    os.makedirs(op_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def mode_label(state: dict | None) -> str | None:
    """scenario + migration_mode 组合标签（与最终报告 vocabulary 一致）。"""
    if not state:
        return None
    scenario = state.get("scenario")
    if not scenario:
        return None
    if scenario == "migration":
        mm = state.get("migration_mode")
        return f"migration-{mm}" if mm else "migration"
    return scenario


def parse_ts(ts):
    try:
        return datetime.strptime(ts, TS_FORMAT)
    except (TypeError, ValueError):
        return None


def read_timeline_events(op_dir: str) -> list:
    """Load timeline events (malformed lines skipped — audit trail stays readable)."""
    path = os.path.join(op_dir, TIMELINE_FILE)
    events: list = []
    if not os.path.exists(path):
        return events
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def stage_attempt_duration(op_dir: str, stage: int, now_str: str) -> float | None:
    """Seconds since the most recent ``start`` event of `stage` (mechanical).

    None when the stage was never started (e.g. direct BLOCKED_* fail) or
    timestamps are unparsable — the field is then omitted from the event.
    """
    start_ts = None
    for ev in read_timeline_events(op_dir):
        if ev.get("action") == "start" and ev.get("stage") == stage:
            ts = ev.get("ts")
            if ts:
                start_ts = ts
    if not start_ts:
        return None
    now_dt, start_dt = parse_ts(now_str), parse_ts(start_ts)
    if now_dt is None or start_dt is None:
        return None
    return round((now_dt - start_dt).total_seconds(), 3)


class Ctx:
    """Shared context for a command run."""

    def __init__(self, args):
        self.op_dir = os.path.abspath(args.dir or os.getcwd())
        self.repo_root = find_repo_root(self.op_dir)
        self.state_path = os.path.join(self.op_dir, STATE_FILE)
        self.state: dict | None = None

    def load(self, errors: list, backfill: bool = True) -> bool:
        if not os.path.exists(self.state_path):
            errors.append(
                err("E-MISSING", f"状态文件不存在：{self.state_path}（先执行 init）")
            )
            return False
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            errors.append(
                err(
                    "E-PARSE",
                    f"状态文件 JSON 损坏：{exc}（用 repair 查看修复建议，或按 timeline 回溯）",
                )
            )
            return False
        if not isinstance(raw, dict):
            errors.append(err("E-PARSE", "状态文件顶层必须是 JSON 对象"))
            return False
        self.state = gate_lint.state_backfill(raw) if backfill else raw
        return True

    def write(self) -> None:
        assert self.state is not None
        self.state["last_updated"] = now_iso()
        atomic_write_json(self.state_path, self.state)

    def timeline(self, action: str, stage=None, verdict=None, **extra) -> None:
        """Append one enriched event (improvement 1.3).

        Field superset per line: ``ts / action / stage / subagent / mode /
        verdict / duration_s`` (+ ``phase_before`` / ``phase_after`` and
        call-site extras kept for compatibility). Enrichment is mechanical:
        subagent via the static stage map, mode copied from the state file,
        duration = now − most recent ``start`` of the same stage.
        """
        ts = now_iso()
        event = {
            "ts": ts,
            "action": action,
            "phase_before": extra.pop("phase_before", None),
            "phase_after": extra.pop("phase_after", None),
        }
        if stage is not None:
            event["stage"] = stage
            event["subagent"] = STAGE_SUBAGENT.get(stage)
            if action in ("complete", "fail"):
                duration = stage_attempt_duration(self.op_dir, stage, ts)
                if duration is not None:
                    event["duration_s"] = duration
        mode = mode_label(self.state)
        if mode:
            event["mode"] = mode
        if verdict is not None:
            event["verdict"] = verdict
        append_timeline(self.op_dir, {**event, **extra})


def artifact_paths(state: dict, ctx: Ctx) -> dict:
    op = state.get("operator_name") or "op"
    base = rel_posix(ctx.repo_root, ctx.op_dir)

    def resolve(field: str, default: str) -> tuple:
        rel = state.get(field) or f"{base}/{default}"
        return rel, os.path.join(ctx.repo_root, rel)

    return {
        "design": resolve("design_md_path", "DESIGN.md"),
        "review": resolve("review_md_path", "REVIEW.md"),
        "kernel": resolve("kernel_py_path", f"{op}.py"),
        "kernel_opt": resolve("kernel_opt_py_path", f"perf_opt/{op}.py"),
        "opt_log": (
            f"{base}/perf_opt/opt_log.md",
            os.path.join(ctx.repo_root, base, "perf_opt", "opt_log.md"),
        ),
        "perf_feedback": (
            f"{base}/perf_opt/perf_feedback.md",
            os.path.join(ctx.repo_root, base, "perf_opt", "perf_feedback.md"),
        ),
        "perf_records": (
            f"{base}/perf_opt/perf_records.jsonl",
            os.path.join(ctx.repo_root, base, "perf_opt", "perf_records.jsonl"),
        ),
    }


def snapshot_stage_artifacts(state: dict, ctx: Ctx, stage: int) -> None:
    """Record SHA256 of the artifacts produced/validated by `stage`."""
    paths = artifact_paths(state, ctx)
    by_stage = {
        1: ["design"],
        2: ["design", "review"],
        3: ["kernel"],
        4: ["kernel_opt", "perf_feedback"],
    }
    hashes = state.setdefault("artifact_hashes", {})
    for key in by_stage.get(stage, []):
        rel, abs_path = paths[key]
        digest = sha256_file(abs_path)
        if digest:
            hashes[rel] = digest


# ---------------------------------------------------------------------------
# gate dispatch
# ---------------------------------------------------------------------------


def run_gate(stage: int, ctx: Ctx, meta_path: str | None, migration_dir: str | None):
    """Run the mechanical gate for `stage`. Returns (failures, warnings, errors).

    `errors` are input problems (missing --meta etc.), not gate failures.
    """
    failures, warnings, errors = [], [], []
    state = ctx.state or {}
    migration = state.get("scenario") == "migration"
    paths = artifact_paths(gate_lint.state_backfill(state), ctx)

    if stage == 0:
        meta = meta_path
        if not meta and migration_dir:
            mstate = os.path.join(migration_dir, MIGRATION_STATE_FILE)
            if os.path.exists(mstate):
                try:
                    with open(mstate, encoding="utf-8") as fh:
                        meta = json.load(fh).get("meta_path")
                except (json.JSONDecodeError, OSError):
                    pass
        if not meta:
            errors.append(
                err("E-GATE-INPUT", "gate 0 需要 --meta（.migration_meta.json 路径）")
            )
        else:
            f, w = gate_lint.meta_lint(meta, ctx.repo_root)
            failures, warnings = failures + f, warnings + w
    elif stage == 1:
        f, w = gate_lint.design_lint(paths["design"][1], migration, ctx.repo_root)
        failures, warnings = failures + f, warnings + w
    elif stage == 2:
        f, w = gate_lint.review_lint(paths["review"][1], migration)
        failures, warnings = failures + f, warnings + w
    elif stage == 3:
        f, w = gate_lint.kernel_lint(
            paths["kernel"][1], state.get("operator_name") or "op"
        )
        failures, warnings = failures + f, warnings + w
    elif stage == 4:
        f, w = gate_lint.perf_lint(
            paths["kernel_opt"][1],
            paths["opt_log"][1],
            paths["perf_feedback"][1],
            paths["perf_records"][1],
            state.get("operator_name") or "op",
            ctx.repo_root,
        )
        failures, warnings = failures + f, warnings + w
    elif stage == 5:
        mstate = None
        if migration_dir:
            mpath = os.path.join(migration_dir, MIGRATION_STATE_FILE)
            if os.path.exists(mpath):
                try:
                    with open(mpath, encoding="utf-8") as fh:
                        mstate = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    mstate = None
        if mstate is None and meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                mstate = {
                    "family": meta.get("family"),
                    "op_slug": meta.get("op_slug"),
                    "functions": {f: {} for f in meta.get("extracted_functions") or []},
                }
            except (json.JSONDecodeError, OSError):
                mstate = None
        if mstate is None:
            errors.append(
                err(
                    "E-GATE-INPUT",
                    "gate 5 需要 --migration-dir（含 .migration_state.json）或 --meta",
                )
            )
        else:
            f, w = gate_lint.integration_lint(mstate, ctx.repo_root)
            failures, warnings = failures + f, warnings + w
    else:
        errors.append(err("E-GATE-INPUT", f"未知 Stage：{stage}"))
    return failures, warnings, errors


def gate_payload(stage: int, failures: list, warnings: list) -> dict:
    return {
        "action": "gate",
        "stage": stage,
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def scenario_stage_plan(scenario: str, migration_mode: str | None, errors: list):
    if scenario == "optimize":
        if migration_mode:
            errors.append(err("E-SCENARIO", "optimize 场景不接受 --migration-mode"))
        return [4]
    if scenario == "migration":
        if migration_mode == "harness":
            # 函数级状态文件：Stage 1–3；Stage 0/5 由 migration 聚合状态跟踪
            return [1, 2, 3]
        if migration_mode == "plain":
            return [1, 2, 3]
        errors.append(
            err("E-SCENARIO", "migration 场景必须指定 --migration-mode harness|plain")
        )
        return []
    if scenario == "new_op":
        if migration_mode:
            errors.append(err("E-SCENARIO", "new_op 场景不接受 --migration-mode"))
        return [1, 2, 3]
    errors.append(err("E-SCENARIO", f"未知 scenario：{scenario}"))
    return []


def cmd_init(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if os.path.exists(ctx.state_path):
        errors.append(
            err(
                "E-EXISTS",
                f"状态文件已存在：{ctx.state_path}（续跑请用 start/complete/fail；损坏用 repair）",
            )
        )
        return emit({"action": "init", "errors": errors}, 1)
    plan = scenario_stage_plan(args.scenario, args.migration_mode, errors)
    if args.stage_plan:
        try:
            plan = [int(x) for x in args.stage_plan.split(",")]
            if not plan or any(s not in STAGE_PHASE for s in plan):
                raise ValueError
        except ValueError:
            errors.append(err("E-SCENARIO", f"非法 --stage-plan：{args.stage_plan}"))
    if errors:
        return emit({"action": "init", "errors": errors}, 1)

    os.makedirs(ctx.op_dir, exist_ok=True)
    base = rel_posix(ctx.repo_root, ctx.op_dir)
    op = args.op
    ts_compact = re.sub(r"[-:]", "", now_iso())
    state = gate_lint.state_backfill(
        {
            "task_id": f"{args.project}-{op}-{ts_compact}",
            "project_name": args.project,
            "operator_name": op,
            "scenario": args.scenario,
            "migration_mode": args.migration_mode,
            "stage_plan": plan,
            "phase": STAGE_PHASE[plan[0]],
            "current_stage": plan[0],
            "user_requirement": args.requirement,
            "design_md_path": args.design_path or f"{base}/DESIGN.md",
            "review_md_path": f"{base}/REVIEW.md",
            "kernel_py_path": args.kernel_path or f"{base}/{op}.py",
            "kernel_opt_py_path": f"{base}/perf_opt/{op}.py",
            "retry_count": 0,
            "max_retry": args.max_retry,
            "final_artifact": None,
            "stage_status": {},
            "stage_retry_count": {str(k): 0 for k in range(6)},
            "stage3_failure_breakdown": {"runtime_fail": 0, "precision_fail": 0},
            "perf_iteration": {
                "count": 0,
                "last_improvement": 0.0,
                "consecutive_no_improvement": 0,
            },
            "perf_tuning_requested": None,
            "env_check_passed": False,
            "failure_reason": None,
            "artifact_hashes": {},
            "last_updated": now_iso(),
        }
    )
    ctx.state = state
    ctx.write()
    ctx.timeline(
        "init",
        phase_after=state["phase"],
        scenario=state["scenario"],
        migration_mode=state["migration_mode"],
        stage_plan=state["stage_plan"],
    )
    return emit(
        {
            "action": "init",
            "state_path": rel_posix(ctx.repo_root, ctx.state_path),
            "phase": state["phase"],
            "stage_plan": state["stage_plan"],
            "note": (
                "harness 函数级状态：Stage 0/5 由 `statectl migration` 聚合状态跟踪"
                if args.scenario == "migration" and args.migration_mode == "harness"
                else None
            ),
        },
        0,
    )


def cmd_start(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "start", "stage": args.stage, "errors": errors}, 1)
    state = ctx.state
    n = args.stage
    phase_before = state["phase"]

    if state["phase"] in TERMINAL_PHASES:
        reopen = (
            args.extend
            and state["phase"] == "DONE"
            and state.get("failure_reason") is None
            and n == 4
            and state.get("scenario") in ("new_op", "migration")
            and state["stage_status"].get("3") == "completed"
            and state.get("perf_tuning_requested") in (None, "yes")
        )
        if not reopen:
            errors.append(
                err(
                    "E-TERMINAL",
                    f"phase={state['phase']} 已终态，禁止 start（Stage 4 可选调优用 --extend 重开）",
                )
            )
            return emit({"action": "start", "stage": n, "errors": errors}, 1)
        state["stage_plan"] = sorted(set(state["stage_plan"]) | {4})
        state["final_artifact"] = None
    elif n not in state["stage_plan"]:
        if args.extend and n == 4:
            state["stage_plan"] = sorted(set(state["stage_plan"]) | {4})
        else:
            errors.append(
                err(
                    "E-NOT-IN-PLAN",
                    f"Stage {n} 不在 stage_plan={state['stage_plan']}（可选 Stage 4 用 --extend 追加）",
                )
            )
            return emit({"action": "start", "stage": n, "errors": errors}, 1)

    others = [
        k
        for k, v in state["stage_status"].items()
        if v == "in_progress" and int(k) != n
    ]
    if others:
        errors.append(
            err(
                "E-CONCURRENT",
                f"Stage {others} 仍为 in_progress，先 complete/fail 再 start {n}（防并发覆盖）",
            )
        )
        return emit({"action": "start", "stage": n, "errors": errors}, 1)

    status = state["stage_status"].get(str(n))
    revision_reentry = (
        status == "completed"
        and n == 1
        and state.get("last_failure_reason") == "design_revision"
        and state["retry_count"] < state["max_retry"]
    )
    if status == "in_progress":
        errors.append(
            err("E-ALREADY-STARTED", f"Stage {n} 已处于 in_progress，禁止重复 start")
        )
        return emit({"action": "start", "stage": n, "errors": errors}, 1)
    if status == "completed" and not revision_reentry:
        errors.append(
            err(
                "E-COMPLETED",
                f"Stage {n} 已 completed，禁止重复进入（设计修订回 Stage 1 须先 fail --reason design_revision）",
            )
        )
        return emit({"action": "start", "stage": n, "errors": errors}, 1)
    if status == "failed":
        limit = STAGE_RETRY_LIMITS.get(n)
        if limit is not None and state["stage_retry_count"].get(str(n), 0) >= limit:
            errors.append(
                err(
                    "E-RETRY-EXCEEDED",
                    f"Stage {n} 重试已达上限 {limit}（状态应已转 FAILED；请核对 failure_reason）",
                )
            )
            return emit({"action": "start", "stage": n, "errors": errors}, 1)
    if status is None:
        for s in state["stage_plan"]:
            if s == n:
                break
            if state["stage_status"].get(str(s)) != "completed":
                errors.append(
                    err(
                        "E-OUT-OF-ORDER",
                        f"禁止跳阶段：Stage {s} 未完成（status="
                        f"{state['stage_status'].get(str(s))}），不能直接 start {n}",
                    )
                )
                return emit({"action": "start", "stage": n, "errors": errors}, 1)

    if revision_reentry:
        state["last_failure_reason"] = None  # 一次性修订通行证，防止无限重入
    state["stage_status"][str(n)] = "in_progress"
    state["phase"] = STAGE_PHASE[n]
    state["current_stage"] = n
    ctx.write()
    budget = budget_watermark(state, ctx.op_dir, now_iso())
    ctx.timeline(
        "start",
        stage=n,
        phase_before=phase_before,
        phase_after=state["phase"],
        **(
            {"budget_exceeded": budget["exceeded"]}
            if budget and budget["exceeded"]
            else {}
        ),
    )
    payload = {"action": "start", "stage": n, "phase": state["phase"]}
    if budget:
        payload["budget"] = budget
    return emit(payload, 0)


def budget_watermark(state: dict, op_dir: str, ts: str) -> dict | None:
    """Mechanical budget water level for an attempted dispatch (E1.3).

    Returns None when no budget dimension is configured (payload unchanged,
    backward compatible). `dispatches_used` counts PRIOR ``start`` events —
    the dispatch being attempted is #used+1, so ``exceeded`` lists dimensions
    this dispatch would blow through. Advisory only: exit code stays 0, the
    conductor routes on ``budget.exceeded`` per gate-and-retry.md §2.
    """
    budget = state.get("budget") or {}
    dims = {
        "max_subagent_dispatches": budget.get("max_subagent_dispatches"),
        "max_wallclock_s": budget.get("max_wallclock_s"),
    }
    if all(v is None for v in dims.values()):
        return None
    events = read_timeline_events(op_dir)
    dispatches = sum(1 for ev in events if ev.get("action") == "start")
    first_ts = next((ev["ts"] for ev in events if ev.get("ts")), None)
    wallclock = None
    if first_ts:
        a, b = parse_ts(first_ts), parse_ts(ts)
        if a and b:
            wallclock = round((b - a).total_seconds(), 3)
    info = {
        "dispatches_used": dispatches,
        "max_subagent_dispatches": dims["max_subagent_dispatches"],
        "wallclock_s": wallclock,
        "max_wallclock_s": dims["max_wallclock_s"],
        "exceeded": [],
    }
    if (
        dims["max_subagent_dispatches"] is not None
        and dispatches >= dims["max_subagent_dispatches"]
    ):
        info["exceeded"].append("max_subagent_dispatches")
    if (
        dims["max_wallclock_s"] is not None
        and wallclock is not None
        and wallclock >= dims["max_wallclock_s"]
    ):
        info["exceeded"].append("max_wallclock_s")
    return info


def choose_final_artifact(state: dict, ctx: Ctx) -> str | None:
    paths = artifact_paths(state, ctx)
    if (
        state["stage_status"].get("4") == "completed"
        or state.get("scenario") == "optimize"
    ):
        rel, abs_path = paths["kernel_opt"]
        if os.path.exists(abs_path):
            return rel
    rel, abs_path = paths["kernel"]
    return rel if os.path.exists(abs_path) else None


def cmd_complete(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "complete", "stage": args.stage, "errors": errors}, 1)
    state = ctx.state
    n = args.stage
    phase_before = state["phase"]

    if state["phase"] in TERMINAL_PHASES:
        errors.append(
            err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 complete")
        )
        return emit({"action": "complete", "stage": n, "errors": errors}, 1)
    status = state["stage_status"].get(str(n))
    if status != "in_progress":
        code = "E-DUPLICATE-COMPLETE" if status == "completed" else "E-NOT-STARTED"
        hint = (
            "该 Stage 已 completed，禁止重复 complete"
            if status == "completed"
            else f"Stage {n} 未 start（status={status}），禁止跳阶段 complete"
        )
        errors.append(err(code, hint))
        return emit({"action": "complete", "stage": n, "errors": errors}, 1)

    # 强制先过门禁：gate 失败则整体失败，不写状态（门禁失败处理流程接管）
    failures, warnings, gate_errors = run_gate(n, ctx, args.meta, args.migration_dir)
    if gate_errors or failures:
        return emit(
            {
                **gate_payload(n, failures, warnings),
                "errors": gate_errors,
                "hint": "门禁失败：按「门禁失败处理流程」执行 statectl fail N 后重试，禁止推进状态",
            },
            1,
        )

    state["stage_status"][str(n)] = "completed"
    next_stage = next(
        (
            s
            for s in state["stage_plan"]
            if state["stage_status"].get(str(s)) != "completed"
        ),
        None,
    )
    if next_stage is None:
        state["phase"] = "DONE"
        state["final_artifact"] = choose_final_artifact(state, ctx)
    else:
        state["phase"] = STAGE_PHASE[next_stage]
        state["current_stage"] = next_stage
    snapshot_stage_artifacts(state, ctx, n)
    ctx.write()
    ctx.timeline(
        "complete",
        stage=n,
        verdict="pass",
        phase_before=phase_before,
        phase_after=state["phase"],
        snapshot=sorted(state.get("artifact_hashes", {})),
    )
    return emit(
        {
            "action": "complete",
            "stage": n,
            "phase": state["phase"],
            "gate": gate_payload(n, [], warnings),
            "final_artifact": state.get("final_artifact"),
        },
        0,
    )


def cmd_fail(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "fail", "stage": args.stage, "errors": errors}, 1)
    state = ctx.state
    n = args.stage
    phase_before = state["phase"]

    if args.blocked_code:
        if args.blocked_code not in gate_lint.BLOCKED_CODES:
            errors.append(err("E-BAD-CODE", f"未知 BLOCKED 码：{args.blocked_code}"))
            return emit({"action": "fail", "stage": n, "errors": errors}, 1)
        state["phase"] = "FAILED"
        state["failure_reason"] = args.blocked_code
        state["last_failure_reason"] = args.blocked_code
        ctx.write()
        ctx.timeline(
            "fail",
            stage=n,
            verdict=args.blocked_code,
            phase_before=phase_before,
            phase_after="FAILED",
            reason=args.blocked_code,
        )
        return emit(
            {
                "action": "fail",
                "stage": n,
                "phase": "FAILED",
                "failure_reason": args.blocked_code,
            },
            0,
        )

    if state["phase"] in TERMINAL_PHASES:
        errors.append(err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 fail"))
        return emit({"action": "fail", "stage": n, "errors": errors}, 1)

    status = state["stage_status"].get(str(n))
    if args.reason != "design_revision":
        if n == 2:
            errors.append(
                err(
                    "E-STAGE2-REVISION",
                    "Stage 2 门禁失败本质是检视不通过，走设计修订循环（--reason design_revision，计 retry_count）",
                )
            )
            return emit({"action": "fail", "stage": n, "errors": errors}, 1)
        if status == "completed":
            errors.append(
                err(
                    "E-COMPLETED-FAIL",
                    f"Stage {n} 已 completed，不能直接 fail（设计修订须用 --reason design_revision）",
                )
            )
            return emit({"action": "fail", "stage": n, "errors": errors}, 1)
        if status not in ("in_progress", "failed"):
            errors.append(
                err(
                    "E-NOT-STARTED",
                    f"Stage {n} 未 start（status={status}），无失败可记",
                )
            )
            return emit({"action": "fail", "stage": n, "errors": errors}, 1)

    if args.reason == "design_revision":
        if status not in ("in_progress", "failed", "completed"):
            errors.append(
                err("E-NOT-STARTED", f"Stage {n} status={status}，不构成设计修订触发源")
            )
            return emit({"action": "fail", "stage": n, "errors": errors}, 1)
        state["stage_status"][str(n)] = "failed"
        state["retry_count"] += 1
        state["last_failure_reason"] = "design_revision"
        # 修订后下游 Stage 全部视为全新实现：状态清空、重试计数清零
        for s in state["stage_plan"]:
            if s > 1:
                state["stage_status"].pop(str(s), None)
                state["stage_retry_count"][str(s)] = 0
        if state["retry_count"] >= state["max_retry"]:
            state["phase"] = "FAILED"
            state["failure_reason"] = "BLOCKED_DESIGN"
            ctx.write()
            ctx.timeline(
                "fail",
                stage=n,
                verdict="design_revision",
                phase_before=phase_before,
                phase_after="FAILED",
                reason="design_revision",
                retry_count=state["retry_count"],
            )
            return emit(
                {
                    "action": "fail",
                    "stage": n,
                    "phase": "FAILED",
                    "failure_reason": "BLOCKED_DESIGN",
                    "retry_count": state["retry_count"],
                },
                0,
            )
        ctx.write()
        ctx.timeline(
            "fail",
            stage=n,
            verdict="design_revision",
            phase_before=phase_before,
            phase_after=state["phase"],
            reason="design_revision",
            retry_count=state["retry_count"],
        )
        return emit(
            {
                "action": "fail",
                "stage": n,
                "reason": "design_revision",
                "retry_count": state["retry_count"],
                "max_retry": state["max_retry"],
                "hint": "修订预算未超限：start 1 重新进入设计（revision 模式）",
            },
            0,
        )

    # 普通 fail：计 stage_retry_count
    state["stage_retry_count"][str(n)] = state["stage_retry_count"].get(str(n), 0) + 1
    state["stage_status"][str(n)] = "failed"
    if n == 3 and args.fail_type in ("runtime", "precision"):
        state["stage3_failure_breakdown"][f"{args.fail_type}_fail"] += 1
    count = state["stage_retry_count"][str(n)]
    limit = STAGE_RETRY_LIMITS.get(n)
    payload = {
        "action": "fail",
        "stage": n,
        "stage_retry_count": count,
        "limit": limit,
    }
    if limit is not None and count >= limit:
        if n == 4:
            # Stage 4 迭代上限：按中止条件交付已验证最优版本（非失败）
            state["phase"] = "DONE"
            state["final_artifact"] = choose_final_artifact(state, ctx)
            payload.update({"phase": "DONE", "abort": "stage4_iteration_limit"})
        else:
            if n == 3:
                reason = (
                    "BLOCKED_ACCURACY"
                    if args.fail_type == "precision"
                    else "BLOCKED_IMPL"
                )
            else:
                reason = BLOCKED_BY_STAGE[n]
            state["phase"] = "FAILED"
            state["failure_reason"] = reason
            payload.update({"phase": "FAILED", "failure_reason": reason})
    ctx.write()
    ctx.timeline(
        "fail",
        stage=n,
        verdict=args.reason or args.fail_type,
        phase_before=phase_before,
        phase_after=state["phase"],
        reason=args.reason or args.fail_type,
        stage_retry_count=count,
    )
    return emit(payload, 0)


def cmd_gate(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    # gate 0/5 可无 stage 状态文件（op 级 / 集成级）；1-4 需要状态文件提供路径
    if args.stage in (0, 5) and not os.path.exists(ctx.state_path):
        ctx.state = {}
    elif not ctx.load(errors):
        return emit({"action": "gate", "stage": args.stage, "errors": errors}, 1)
    failures, warnings, gate_errors = run_gate(
        args.stage, ctx, args.meta, args.migration_dir
    )
    if gate_errors:
        return emit(
            {**gate_payload(args.stage, failures, warnings), "errors": gate_errors}, 1
        )
    return emit(gate_payload(args.stage, failures, warnings), 0 if not failures else 1)


def cmd_snapshot(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "snapshot", "errors": errors}, 1)
    for stage in (1, 2, 3, 4):
        snapshot_stage_artifacts(ctx.state, ctx, stage)
    ctx.write()
    ctx.timeline("snapshot", phase_after=ctx.state["phase"])
    return emit(
        {"action": "snapshot", "artifact_hashes": ctx.state["artifact_hashes"]}, 0
    )


def cmd_verify(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if not ctx.load(errors, backfill=False):
        return emit({"action": "verify", "errors": errors}, 1)
    # schema 检查针对原始内容（backfill 会掩盖缺字段），其余逻辑用补齐后的状态
    problems, _schema_warnings = gate_lint.state_schema_lint(ctx.state, ctx.state_path)
    state = gate_lint.state_backfill(ctx.state)
    ctx.state = state
    drift = []
    for rel, digest in (state.get("artifact_hashes") or {}).items():
        abs_path = os.path.join(ctx.repo_root, rel)
        current = sha256_file(abs_path)
        if current is None:
            drift.append({"path": rel, "issue": "工件缺失（已被删除/移动）"})
        elif current != digest:
            drift.append(
                {
                    "path": rel,
                    "issue": "工件漂移：内容与该 Stage 完成时快照不一致，建议回退该 Stage 重新验证",
                }
            )

    paths = artifact_paths(state, ctx)
    consistency = []
    for stage, key in ((1, "design"), (2, "review"), (3, "kernel"), (4, "kernel_opt")):
        if state["stage_status"].get(str(stage)) == "completed":
            rel, abs_path = paths[key]
            if not os.path.exists(abs_path):
                consistency.append(
                    {
                        "stage": stage,
                        "path": rel,
                        "issue": "Stage 已 completed 但工件不存在",
                    }
                )
    ok = not (problems or drift or consistency)
    if ok:
        ctx.timeline("verify", phase_after=state["phase"], verdict="ok")
    return emit(
        {
            "action": "verify",
            "schema_failures": problems,
            "drift": drift,
            "consistency": consistency,
        },
        0 if ok else 1,
    )


def cmd_repair(args) -> int:
    ctx = Ctx(args)
    if not os.path.exists(ctx.state_path):
        return emit(
            {
                "action": "repair",
                "errors": [err("E-MISSING", f"状态文件不存在：{ctx.state_path}")],
            },
            1,
        )
    parse_error = None
    try:
        with open(ctx.state_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        parse_ok = isinstance(raw, dict)
    except json.JSONDecodeError as exc:
        raw, parse_ok = None, False
        parse_error = str(exc)
    if not parse_ok:
        suggestion = (
            "状态文件损坏且无法解析：① 按 .task_timeline.jsonl 回溯最后合法状态手工重建；"
            "② 或备份后重新 init（会丢失计数，须人工核对磁盘工件实际进度）"
        )
        return emit(
            {
                "action": "repair",
                "ok": False,
                "errors": [
                    err(
                        "E-PARSE",
                        f"JSON 损坏：{parse_error or '顶层不是 JSON 对象'}",
                    )
                ],
                "suggestion": suggestion,
            },
            1,
        )

    before_keys = set(raw.keys())
    state = gate_lint.state_backfill(raw)
    schema_failures, _schema_warnings = gate_lint.state_schema_lint(
        state, ctx.state_path
    )
    fixes = sorted(set(state.keys()) - before_keys)
    # 篡改/漂移检测只报告，不自动修
    drift = []
    for rel, digest in (state.get("artifact_hashes") or {}).items():
        abs_path = os.path.join(ctx.repo_root, rel)
        current = sha256_file(abs_path)
        if current is not None and current != digest:
            drift.append(rel)
    suggestions = []
    if drift:
        suggestions.append(
            f"工件漂移：{drift} —— 建议回退对应 Stage 重新验证（勿盲改哈希）"
        )
    in_progress = [k for k, v in state["stage_status"].items() if v == "in_progress"]
    if len(in_progress) > 1:
        suggestions.append(
            f"多个 Stage 同时 in_progress：{in_progress} —— 按 timeline 判定真实阶段，其余置 failed"
        )
    if fixes and args.apply:
        ctx.state = state
        ctx.write()
        ctx.timeline("repair", phase_after=state["phase"], applied=fixes)
    return emit(
        {
            "action": "repair",
            "applied": fixes if args.apply else [],
            "would_fix": fixes if not args.apply else [],
            "schema_failures": schema_failures,
            "drift": drift,
            "suggestions": suggestions,
            "hint": "重跑以应用补齐：statectl repair --apply",
        },
        0,
    )


def cmd_set(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "set", "errors": errors}, 1)
    state = ctx.state
    if args.perf_tuning is not None:
        if args.perf_tuning not in ("yes", "no"):
            errors.append(err("E-VALUE", "--perf-tuning 只接受 yes|no"))
        else:
            state["perf_tuning_requested"] = args.perf_tuning
    if args.final_artifact is not None:
        state["final_artifact"] = args.final_artifact
    if args.requirement is not None:
        state["user_requirement"] = args.requirement
    if args.env_check is not None:
        if args.env_check not in ("true", "false"):
            errors.append(err("E-VALUE", "--env-check 只接受 true|false"))
        else:
            state["env_check_passed"] = args.env_check == "true"
    if args.perf_feedback_action is not None:
        rel, abs_path = artifact_paths(state, ctx)["perf_feedback"]
        if not os.path.exists(abs_path):
            errors.append(
                err(
                    "E-MISSING",
                    f"perf_feedback.md 不存在：{rel}（[DESIGN_LIMIT] 工件须先产出再记录路由）",
                )
            )
        else:
            state["perf_feedback"] = {
                "path": rel,
                "action": args.perf_feedback_action,
            }
    # A2: perf_iteration 三字段（Stage 4 迭代计数——count / last_improvement /
    # consecutive_no_improvement，schema 见 _shared/standards/signal-registry.md §4）
    if (
        args.perf_iteration_count is not None
        or args.perf_iteration_last_improvement is not None
        or args.perf_iteration_no_improve is not None
    ):
        perf_iter = state.get("perf_iteration")
        if not isinstance(perf_iter, dict):
            perf_iter = {
                "count": 0,
                "last_improvement": 0.0,
                "consecutive_no_improvement": 0,
            }
        if args.perf_iteration_count is not None:
            if args.perf_iteration_count < 0:
                errors.append(err("E-VALUE", "--perf-iteration-count 须为非负整数"))
            else:
                perf_iter["count"] = args.perf_iteration_count
        if args.perf_iteration_last_improvement is not None:
            perf_iter["last_improvement"] = args.perf_iteration_last_improvement
        if args.perf_iteration_no_improve is not None:
            if args.perf_iteration_no_improve < 0:
                errors.append(
                    err("E-VALUE", "--perf-iteration-no-improve 须为非负整数")
                )
            else:
                perf_iter["consecutive_no_improvement"] = args.perf_iteration_no_improve
        if not any(
            e["code"] == "E-VALUE" and "perf-iteration" in e["message"] for e in errors
        ):
            state["perf_iteration"] = perf_iter
    # E1.3: 任务级预算（max_wallclock_s / max_subagent_dispatches /
    # max_stage4_experiments；正数或 null=不限，默认见 gate-and-retry.md §2）
    if args.budget_json is not None:
        budget_errors: list = []
        try:
            budget = json.loads(args.budget_json)
        except json.JSONDecodeError as exc:
            budget_errors.append(err("E-VALUE", f"--budget-json 非法 JSON：{exc}"))
            budget = None
        allowed = (
            "max_wallclock_s",
            "max_subagent_dispatches",
            "max_stage4_experiments",
        )
        if isinstance(budget, dict):
            unknown = sorted(set(budget) - set(allowed))
            if unknown:
                budget_errors.append(
                    err(
                        "E-VALUE",
                        f"--budget-json 含未知字段：{unknown}（允许：{list(allowed)}）",
                    )
                )
            else:
                for key, val in budget.items():
                    if val is None:
                        continue
                    if (
                        isinstance(val, bool)
                        or not isinstance(val, (int, float))
                        or val <= 0
                    ):
                        budget_errors.append(
                            err("E-VALUE", f"budget.{key} 须为正数或 null（不限）")
                        )
                if not budget_errors:
                    merged = dict(state.get("budget") or {})
                    merged.update(budget)
                    state["budget"] = merged
        elif budget is not None:
            budget_errors.append(err("E-VALUE", "--budget-json 须为 JSON 对象"))
        errors.extend(budget_errors)
    if errors:
        return emit({"action": "set", "errors": errors}, 1)
    ctx.write()
    extra = {}
    if state.get("perf_feedback"):
        extra["perf_feedback_action"] = state["perf_feedback"].get("action")
    ctx.timeline("set", phase_after=state["phase"], **extra)
    return emit({"action": "set", "state": state}, 0)


def cmd_show(args) -> int:
    ctx = Ctx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "show", "errors": errors}, 1)
    return emit({"action": "show", "state": ctx.state}, 0)


def cmd_timeline_summary(args) -> int:
    """Mechanical summary of .task_timeline.jsonl (improvement 1.3).

    Read-only (no state file needed, no event appended) — final report
    「时间线」段的数据源 + evolver 失败根因链一手输入。Works on both
    stage timelines and migration aggregate timelines.
    """
    ctx = Ctx(args)
    path = os.path.join(ctx.op_dir, TIMELINE_FILE)
    if not os.path.exists(path):
        return emit(
            {
                "action": "timeline-summary",
                "errors": [
                    err(
                        "E-MISSING",
                        f"时间线文件不存在：{path}（statectl 首次状态迁移后自动生成）",
                    )
                ],
            },
            1,
        )
    events = read_timeline_events(ctx.op_dir)
    action_counts: dict = {}
    stages: dict = {}
    failures: list = []
    ts_list: list = []
    mode = None
    for ev in events:
        action = ev.get("action") or "?"
        action_counts[action] = action_counts.get(action, 0) + 1
        ts = ev.get("ts")
        if ts:
            ts_list.append(ts)
        if not mode and ev.get("mode"):
            mode = ev["mode"]
        stage = ev.get("stage")
        if stage is None:
            continue
        rec = stages.setdefault(
            stage,
            {
                "stage": stage,
                "subagent": ev.get("subagent") or STAGE_SUBAGENT.get(stage),
                "attempts": 0,
                "completed": 0,
                "failed": 0,
                "durations_s": [],
            },
        )
        if action == "start":
            rec["attempts"] += 1
        elif action == "complete":
            rec["completed"] += 1
            if ev.get("duration_s") is not None:
                rec["durations_s"].append(ev["duration_s"])
        elif action == "fail":
            rec["failed"] += 1
            if ev.get("duration_s") is not None:
                rec["durations_s"].append(ev["duration_s"])
            failures.append(
                {
                    "stage": stage,
                    "subagent": rec["subagent"],
                    "verdict": ev.get("verdict") or ev.get("reason"),
                    "ts": ts,
                    "duration_s": ev.get("duration_s"),
                }
            )
    first_ts = ts_list[0] if ts_list else None
    last_ts = ts_list[-1] if ts_list else None
    elapsed = None
    if first_ts and last_ts:
        a, b = parse_ts(first_ts), parse_ts(last_ts)
        if a and b:
            elapsed = round((b - a).total_seconds(), 3)
    stage_rows = []
    for stage in sorted(stages):
        rec = stages[stage]
        durations = rec.pop("durations_s")
        rec["duration_s_total"] = round(sum(durations), 3) if durations else None
        rec["durations_s"] = durations
        stage_rows.append(rec)
    return emit(
        {
            "action": "timeline-summary",
            "timeline_path": rel_posix(ctx.repo_root, path),
            "events_total": len(events),
            "mode": mode,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "elapsed_s": elapsed,
            "dispatches": action_counts.get("start", 0),
            "action_counts": action_counts,
            "stages": stage_rows,
            "failures": failures,
        },
        0,
    )


# ---------------------------------------------------------------------------
# migration aggregate state (.migration_state.json)
# ---------------------------------------------------------------------------


class MCtx(Ctx):
    """Migration aggregate context — same mechanics, different state file and
    NO stage-state schema backfill (aggregate has its own schema)."""

    def __init__(self, args):
        self.op_dir = os.path.abspath(args.dir or os.getcwd())
        self.repo_root = find_repo_root(self.op_dir)
        self.state_path = os.path.join(self.op_dir, MIGRATION_STATE_FILE)
        self.state: dict | None = None

    def load(self, errors: list, backfill: bool = False) -> bool:
        return super().load(errors, backfill=False)


def cmd_migration_init(args) -> int:
    ctx = MCtx(args)
    if os.path.exists(ctx.state_path):
        return emit(
            {
                "action": "migration-init",
                "errors": [err("E-EXISTS", f"聚合状态已存在：{ctx.state_path}")],
            },
            1,
        )
    phase = args.phase
    if phase not in ("SCAFFOLD", "DEV_LOOP"):
        return emit(
            {
                "action": "migration-init",
                "errors": [err("E-PHASE-ORDER", "init 只接受 SCAFFOLD|DEV_LOOP")],
            },
            1,
        )
    os.makedirs(ctx.op_dir, exist_ok=True)
    funcs = {}
    for f in args.functions:
        funcs[f] = {"stage_state": None, "status": "pending"}
    state = {
        "op_name": args.op_slug,
        "op_slug": args.op_slug,
        "family": args.family,
        "meta_path": args.meta_path,
        "phase": phase,
        "functions": funcs,
        "integration": {"attempts": 0, "status": None},
        "last_updated": now_iso(),
    }
    ctx.state = state
    ctx.write()
    ctx.timeline("migration-init", phase_after=phase, functions=list(funcs))
    return emit(
        {"action": "migration-init", "phase": phase, "functions": list(funcs)}, 0
    )


def cmd_migration_func(args) -> int:
    ctx = MCtx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "migration-func", "errors": errors}, 1)
    state = ctx.state
    if args.name not in state.get("functions", {}):
        errors.append(
            err(
                "E-FUNC-UNKNOWN",
                f"函数 {args.name} 不在 functions（{list(state['functions'])}）",
            )
        )
        return emit({"action": "migration-func", "errors": errors}, 1)
    if state["phase"] not in ("SCAFFOLD", "DEV_LOOP"):
        errors.append(
            err("E-PHASE-ORDER", f"phase={state['phase']} 下不允许更新函数状态")
        )
        return emit({"action": "migration-func", "errors": errors}, 1)
    if args.status not in FUNC_STATUSES:
        errors.append(err("E-VALUE", f"非法函数状态：{args.status}"))
        return emit({"action": "migration-func", "errors": errors}, 1)
    state["functions"][args.name]["status"] = args.status
    ctx.write()
    ctx.timeline(
        "migration-func", func=args.name, status=args.status, phase_after=state["phase"]
    )
    return emit(
        {"action": "migration-func", "func": args.name, "status": args.status}, 0
    )


def cmd_migration_phase(args) -> int:
    ctx = MCtx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "migration-phase", "errors": errors}, 1)
    state = ctx.state
    target = args.phase
    current = state["phase"]
    if current in ("DONE", "FAILED"):
        errors.append(err("E-TERMINAL", f"聚合状态已终态：{current}"))
        return emit({"action": "migration-phase", "errors": errors}, 1)
    order = ["SCAFFOLD", "DEV_LOOP", "INTEGRATE", "DONE"]
    if target == "FAILED":
        if not args.reason:
            errors.append(err("E-VALUE", "FAILED 需要 --reason（BLOCKED_* 码或原因）"))
            return emit({"action": "migration-phase", "errors": errors}, 1)
    elif target not in order or (
        target in order and order.index(target) <= order.index(current)
    ):
        errors.append(
            err(
                "E-PHASE-ORDER",
                f"非法迁移：{current} → {target}（合法序：{' → '.join(order)}）",
            )
        )
        return emit({"action": "migration-phase", "errors": errors}, 1)
    if target == "INTEGRATE":
        pending = [f for f, v in state["functions"].items() if v["status"] != "done"]
        if pending:
            errors.append(
                err(
                    "E-PHASE-PRECOND",
                    f"函数未全部 done，不能集成：{pending}",
                )
            )
            return emit({"action": "migration-phase", "errors": errors}, 1)
    if target == "DONE" and state["integration"].get("status") != "pass":
        errors.append(
            err(
                "E-PHASE-PRECOND",
                "integration.status != pass：先 `migration integrate --status pass`（pytest 通过后）",
            )
        )
        return emit({"action": "migration-phase", "errors": errors}, 1)
    state["phase"] = target
    if target == "FAILED":
        state["failure_reason"] = args.reason
    ctx.write()
    ctx.timeline(
        "migration-phase", phase_before=current, phase_after=target, reason=args.reason
    )
    return emit({"action": "migration-phase", "phase": target}, 0)


def cmd_migration_integrate(args) -> int:
    ctx = MCtx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "migration-integrate", "errors": errors}, 1)
    state = ctx.state
    if state["phase"] != "INTEGRATE":
        errors.append(
            err("E-PHASE-ORDER", f"phase={state['phase']}，须先迁移到 INTEGRATE")
        )
        return emit({"action": "migration-integrate", "errors": errors}, 1)
    if args.status not in ("pass", "fail"):
        errors.append(err("E-VALUE", "--status 只接受 pass|fail"))
        return emit({"action": "migration-integrate", "errors": errors}, 1)
    if args.status == "pass":
        state["integration"]["status"] = "pass"
        ctx.write()
        ctx.timeline("migration-integrate", status="pass", phase_after=state["phase"])
        return emit(
            {
                "action": "migration-integrate",
                "status": "pass",
                "hint": "可执行 `migration phase --phase DONE`",
            },
            0,
        )
    state["integration"]["attempts"] += 1
    attempts = state["integration"]["attempts"]
    payload = {"action": "migration-integrate", "status": "fail", "attempts": attempts}
    if attempts > 2:
        state["phase"] = "FAILED"
        state["failure_reason"] = "BLOCKED_INTEGRATION"
        payload.update({"phase": "FAILED", "failure_reason": "BLOCKED_INTEGRATION"})
    ctx.write()
    ctx.timeline(
        "migration-integrate",
        status="fail",
        attempts=attempts,
        phase_after=state["phase"],
    )
    return emit(payload, 0)


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="statectl",
        description="tilelang-op-conductor 确定性状态机 CLI"
        "（U1 修复：转换校验/计数/门禁/哈希；1.3：timeline 事件流 + timeline-summary）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_dir(sp, required=True):
        sp.add_argument(
            "--dir", required=required, help="算子目录（含 .stage_state.json）"
        )

    sp = sub.add_parser("init", help="按场景路由表生成合法初始状态")
    add_dir(sp)
    sp.add_argument("--project", required=True)
    sp.add_argument("--op", required=True)
    sp.add_argument(
        "--scenario", required=True, choices=["new_op", "migration", "optimize"]
    )
    sp.add_argument("--migration-mode", choices=["harness", "plain"])
    sp.add_argument("--requirement", default=None)
    sp.add_argument("--max-retry", type=int, default=3)
    sp.add_argument("--stage-plan", default=None, help="覆盖默认 plan（如 0,1,2,3,5）")
    sp.add_argument("--design-path", default=None)
    sp.add_argument("--kernel-path", default=None, help="optimize 场景指向既有 kernel")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("start", help="start_stage(N)：校验转换合法性后置 in_progress")
    add_dir(sp)
    sp.add_argument("stage", type=int, choices=range(6))
    sp.add_argument(
        "--extend",
        action="store_true",
        help="把可选 Stage 4 追加进 stage_plan（含 DONE 后重开）",
    )
    sp.set_defaults(func=cmd_start)

    sp = sub.add_parser("complete", help="complete_stage(N)：强制先过 gate，通过才落盘")
    add_dir(sp)
    sp.add_argument("stage", type=int, choices=range(6))
    sp.add_argument("--meta", default=None, help="gate 0 用：.migration_meta.json 路径")
    sp.add_argument(
        "--migration-dir",
        default=None,
        help="gate 5 用：含 .migration_state.json 的 op 级目录",
    )
    sp.set_defaults(func=cmd_complete)

    sp = sub.add_parser("fail", help="fail_stage(N)：计数、上限判定与 BLOCKED_* 路由")
    add_dir(sp)
    sp.add_argument("stage", type=int, choices=range(6))
    sp.add_argument("--reason", default=None, help="design_revision 走设计修订预算")
    sp.add_argument("--fail-type", choices=["runtime", "precision"], default="runtime")
    sp.add_argument(
        "--blocked-code",
        default=None,
        help="直接终态码（BLOCKED_SPEC/BLOCKED_ENVIRONMENT 等，不耗重试）",
    )
    sp.set_defaults(func=cmd_fail)

    sp = sub.add_parser("gate", help="单独执行 Stage N 机械门禁（JSON 输出，不改状态）")
    add_dir(sp, required=False)
    sp.add_argument("stage", type=int, choices=range(6))
    sp.add_argument("--meta", default=None)
    sp.add_argument("--migration-dir", default=None)
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("snapshot", help="对现存工件记录 SHA256 快照")
    add_dir(sp)
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("verify", help="schema + 哈希漂移 + 阶段-工件一致性校验")
    add_dir(sp)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("repair", help="检测状态-工件不一致并给修复建议")
    add_dir(sp)
    sp.add_argument("--apply", action="store_true", help="应用安全修复（schema 补齐）")
    sp.set_defaults(func=cmd_repair)

    sp = sub.add_parser("set", help="安全写入少量白名单字段")
    add_dir(sp)
    sp.add_argument("--perf-tuning", default=None, choices=["yes", "no"])
    sp.add_argument("--final-artifact", default=None)
    sp.add_argument("--requirement", default=None)
    sp.add_argument("--env-check", default=None, choices=["true", "false"])
    sp.add_argument(
        "--perf-feedback-action",
        default=None,
        choices=["archive", "revise", "none"],
        help="记录 [DESIGN_LIMIT] 逆向反馈路由（附录补记/设计修订/不处理）",
    )
    sp.add_argument(
        "--perf-iteration-count",
        type=int,
        default=None,
        help="perf_iteration.count：已完成调优轮数（A2）",
    )
    sp.add_argument(
        "--perf-iteration-last-improvement",
        type=float,
        default=None,
        help="perf_iteration.last_improvement：最近一次提升幅度（%%，A2）",
    )
    sp.add_argument(
        "--perf-iteration-no-improve",
        type=int,
        default=None,
        help="perf_iteration.consecutive_no_improvement：连续无有效提升轮数（A2）",
    )
    sp.add_argument(
        "--budget-json",
        default=None,
        help="任务级预算 JSON，如 '{\"max_subagent_dispatches\": 40}'"
        "（字段：max_wallclock_s / max_subagent_dispatches / max_stage4_experiments；E1.3）",
    )
    sp.set_defaults(func=cmd_set)

    sp = sub.add_parser("show", help="打印当前状态 JSON")
    add_dir(sp)
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser(
        "timeline-summary",
        help="机械汇总 .task_timeline.jsonl（各 Stage attempts/耗时/失败链——"
        "最终报告「时间线」段与 evolver 根因链的数据源，只读）",
    )
    add_dir(sp)
    sp.set_defaults(func=cmd_timeline_summary)

    m = sub.add_parser("migration", help="harness 聚合状态 .migration_state.json")
    msub = m.add_subparsers(dest="migration_command", required=True)

    sp = msub.add_parser("init")
    sp.add_argument("--dir", required=True, help="op 级目录 examples/{op_slug}")
    sp.add_argument("--op-slug", required=True)
    sp.add_argument("--family", required=True)
    sp.add_argument("--meta-path", required=True)
    sp.add_argument("--functions", nargs="+", required=True)
    sp.add_argument("--phase", default="DEV_LOOP", choices=["SCAFFOLD", "DEV_LOOP"])
    sp.set_defaults(func=cmd_migration_init)

    sp = msub.add_parser("func", help="更新函数状态")
    sp.add_argument("--dir", required=True)
    sp.add_argument("name")
    sp.add_argument("--status", required=True, choices=sorted(FUNC_STATUSES))
    sp.set_defaults(func=cmd_migration_func)

    sp = msub.add_parser(
        "phase", help="聚合 phase 迁移（SCAFFOLD→DEV_LOOP→INTEGRATE→DONE）"
    )
    sp.add_argument("--dir", required=True)
    sp.add_argument("--phase", required=True, choices=MIG_PHASES)
    sp.add_argument("--reason", default=None)
    sp.set_defaults(func=cmd_migration_phase)

    sp = msub.add_parser(
        "integrate", help="记录 Stage 5 集成验证结果（pass/fail 计次）"
    )
    sp.add_argument("--dir", required=True)
    sp.add_argument("--status", required=True, choices=["pass", "fail"])
    sp.add_argument("--reason", default=None)
    sp.set_defaults(func=cmd_migration_integrate)

    sp = msub.add_parser("show")
    sp.add_argument("--dir", required=True)
    sp.set_defaults(func=cmd_migration_show)

    return p


def cmd_migration_show(args) -> int:
    ctx = MCtx(args)
    errors: list = []
    if not ctx.load(errors):
        return emit({"action": "migration-show", "errors": errors}, 1)
    return emit({"action": "migration-show", "state": ctx.state}, 0)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.error("missing command")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
