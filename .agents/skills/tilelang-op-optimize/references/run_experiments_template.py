"""run_experiments.py -- Stage 4 experiment batch runner template (T-4).

Copied into perf_opt/ by the optimizer on its FIRST Phase 2 round and adapted
(this file is a TEMPLATE -- the adaptation points are the CONFIG constants
and the `build_bench_cmd` function below). After the first round, each round
just updates EXPERIMENTS and re-runs this script; the agent only reads the
summary table it prints. This turns "agent round-trips drive experiments"
into "script executes, agent reads the table" (>=60% fewer agent interaction
rounds; also immune to the large-artifact session-overflow failure class,
see queue VP-2026-0021).

Per experiment branch this runner, in order:
  1. runs the L0 precision regression (branch file `--level L0`);
  2. wraps the msprof op measurement (kernel-only Task Duration, the single
     latency metric);
  3. appends one line per measured tune workload to perf_records.jsonl (append-only, contract:
     _shared/standards/signal-registry.md #5);
  4. prints a candidate-vs-current-best summary table at the end.

Adaptation points (marked "ADAPT"):
  - CONFIG: kernel identity and direct kernel invocation; tune workloads are read
    from workload_inventory.json, which must come from the migrated benchmark;
  - build_bench_cmd(): how a branch file is launched for msprof profiling;
  - EXPERIMENTS: the round's branch list (file + opt_id + parent_id).

Usage (inside perf_opt/):
  python run_experiments.py --round 3 --workload case_id  # candidate: one tune workload
  python run_experiments.py --phase baseline --all        # baseline: every tune workload
  python run_experiments.py --phase final --all  # final: every tune workload
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys

# --------------------------------------------------------------- CONFIG (ADAPT)
OP = "my_op"  # ADAPT: operator name (file stem)
KERNEL_ID = "path/to/kernel.py::factory::main"  # ADAPT: unique implementation identity
KERNEL_NAME = "main"  # ADAPT: target kernel name for msprof
DIRECT_CMD = [sys.executable, "{impl}", "--case", "{workload_id}"]  # ADAPT: direct kernel argv; never the Stage 5 wrapper benchmark
LAUNCH_COUNT = 15
WARM_UP = 5
TIMEOUT_S = 900
CURRENT_BEST = f"{OP}.py"  # ADAPT: round base (current best file)
CURRENT_BEST_ID = "baseline"  # ADAPT: ID of the current merged kernel version
# --------------------------------------------------------------- /CONFIG

PERF_RECORDS = "perf_records.jsonl"
WORKLOAD_INVENTORY = "workload_inventory.json"
MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)

# ----------------------------------------------------------- EXPERIMENTS (ADAPT)
# Each entry: {file, opt_id, parent_id}
EXPERIMENTS = [
    # Baseline: {"file": f"{OP}.py", "opt_id": "baseline", "parent_id": None},
    # Final: {"file": f"{OP}.py", "opt_id": "merged-best", "parent_id": "previous-best"},
    # {"file": f"{OP}_round2_candidate1.py", "opt_id": "round2_candidate1",
    #  "parent_id": "baseline"},
]
# ------------------------------------------------------------------ /EXPERIMENTS


def build_bench_cmd(impl_path: str, workload_id: str) -> list[str]:  # ADAPT if direct kernel entry needs more args
    return [part.format(impl=impl_path, workload_id=workload_id) for part in DIRECT_CMD]


def safe_component(value: str) -> str:
    """Keep benchmark IDs out of path traversal and avoid sanitized collisions."""
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:64] or "case"
    return f"{label}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"


def run_l0(branch_file: str) -> bool:
    proc = subprocess.run(
        [sys.executable, branch_file, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    return proc.returncode == 0


def run_msprof(branch_file: str, workload_id: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "posix":
        os.chmod(out_dir, 0o700)
    cmd = [
        "msprof", "op", f"--kernel-name={KERNEL_NAME}", f"--output={out_dir}",
        f"--launch-count={LAUNCH_COUNT}", f"--warm-up={WARM_UP}", "--dump=off",
        f"--aic-metrics={MSPROF_METRICS}", *build_bench_cmd(branch_file, workload_id),
    ]
    log = os.path.join("logs", safe_component(out_dir) + ".log")
    os.makedirs("logs", exist_ok=True)
    with open(log, "w") as f:
        f.write(f"# {shlex.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_S
        )
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"error": f"msprof rc={proc.returncode}", "log": log}
    durs = []
    for path in sorted(
        glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    ):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    if not durs:
        return {"error": "no OpBasicInfo rows", "log": log}
    return {
        "duration_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "msprof_raw_path": out_dir,
    }


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tune_workload_ids() -> list[str]:
    with open(WORKLOAD_INVENTORY, encoding="utf-8") as handle:
        inventory = json.load(handle)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("workloads"), list):
        raise ValueError("workload_inventory.json must contain a workloads array")
    return [
        item["workload_id"] for item in inventory["workloads"]
        if item.get("kernel_id") == KERNEL_ID and item.get("kind") == "tune"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int)
    ap.add_argument("--only", help="run a single opt_id")
    ap.add_argument("--workload", help="one non-smoke benchmark workload ID")
    ap.add_argument("--all", action="store_true", help="measure every tune workload (baseline/final/merged)")
    ap.add_argument("--phase", choices=("baseline", "candidate", "merged", "final"), default="candidate")
    args = ap.parse_args()

    if args.round is None:
        args.round = 0 if args.phase == "baseline" else 1
    if args.phase == "baseline" and args.round != 0:
        ap.error("baseline measurements must use round 0")

    if args.phase == "candidate":
        if args.all or not args.workload:
            ap.error("candidate measurements require exactly one --workload")
    elif not args.all or args.workload:
        ap.error("baseline/merged/final measurements require --all and no --workload")
    available = tune_workload_ids()
    selected = list(available) if args.all else [args.workload]
    unknown = [workload_id for workload_id in selected if workload_id not in available]
    if unknown or not selected:
        ap.error(f"unknown or empty tune workload selection: {unknown or selected}")

    todo = [e for e in EXPERIMENTS if not args.only or e["opt_id"] == args.only]
    if not todo:
        print("no experiments configured (edit EXPERIMENTS)")
        return 2

    if args.phase in ("baseline", "merged", "final") and len(todo) != 1:
        ap.error("baseline/merged/final measurements must use exactly one candidate per kernel")
    results = []
    for exp in todo:
        branch = exp["file"]
        print(f"=== {exp['opt_id']}: {branch} ===")
        if not os.path.isfile(branch):
            results.append({**exp, "status": "missing_file"})
            continue
        l0 = run_l0(branch)
        if not l0:
            results.append({**exp, "status": "l0_fail", "l0_pass": False})
            continue
        for workload_id in selected:
            prof = run_msprof(
                branch, workload_id,
                os.path.join("profiles", args.phase, f"round{args.round}", safe_component(exp["opt_id"]), safe_component(workload_id)),
            )
            if "error" in prof:
                results.append({**exp, "workload_id": workload_id, "status": "msprof_fail", "l0_pass": True})
                print(f"  {workload_id}: msprof FAILED: {prof['error']}")
                continue
            append_record(
                {
                    "round": args.round,
                    "candidate_id": exp["opt_id"],
                    "parent_id": exp.get("parent_id"),
                    "kernel_id": KERNEL_ID,
                    "workload_id": workload_id,
                    "phase": args.phase,
                    "artifact_path": branch,
                    "artifact_sha256": file_sha256(branch),
                    "duration_us": prof["duration_us"],
                    "l0_pass": True,
                    "msprof_raw_path": prof["msprof_raw_path"],
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
            results.append(
                {**exp, "workload_id": workload_id, "status": "ok", "l0_pass": True, "duration_us": prof["duration_us"]}
            )
            print(f"  {workload_id}: L0 pass, median Task Duration = {prof['duration_us']} us")

    # summary table: candidates vs current best (B2 structured backflow)
    bases = {}
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if r.get("kernel_id") == KERNEL_ID and r.get("candidate_id") == CURRENT_BEST_ID:
                bases[r.get("workload_id")] = r["duration_us"]
    print("\n| branch | workload_id | l0 | Task Duration(us) | vs current best | status |")
    print("|---|---|---|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        rel = ""
        base = bases.get(r.get("workload_id"))
        if dur is not None and base:
            rel = f"{(base - dur) / base * 100:+.1f}%"
        print(
            f"| {r['opt_id']} | {r.get('workload_id', '-')} | {r.get('l0_pass', '-')} | "
            f"{dur if dur is not None else '-'} | {rel} | {r['status']} |"
        )
    print(f"\n(current merged best candidate: {CURRENT_BEST_ID})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
