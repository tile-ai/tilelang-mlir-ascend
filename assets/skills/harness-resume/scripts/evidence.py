"""Versioned, deterministic evidence checks. No device execution or approval claims."""

import hashlib
import json
import math
from pathlib import Path
import statistics


from projectctl import Invalid

METRICS = {"cpu": "cpu_wall_ms", "torch_npu": "mean_device_busy_union_ms"}


def require(condition, message):
    if not condition:
        raise Invalid(message)


def number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def evaluator_identity():
    root = Path(__file__).resolve().parents[2]
    paths = [*Path(__file__).parent.glob('*.py'),
             * (root / 'harness-develop/scripts').glob('*.py'),
             root / 'harness-new-project/scripts/records.py',
             root / 'harness-new-project/scripts/projectctl.py']
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def bundle_identity():
    root = Path(__file__).resolve().parents[2]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for directory in sorted(root.glob("harness-*")) for path in sorted(directory.rglob("*"))
              if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def freeze(store, ref, dependencies):
    if ref["path"] in dependencies:
        require(dependencies[ref["path"]] == ref, "conflicting artifact identity")
        return
    store.verify_ref(ref)
    dependencies[ref["path"]] = ref
    path = store.local(ref["path"])
    if path.suffix != ".json":
        return
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        return  # Opaque artifacts are hashed; only the explicit index schema is traversed.
    if isinstance(data, dict) and str(data.get("schema", "")).startswith(("operator-harness/artifact-index@", "operator-harness/diagnostic-evidence@")):
        require(data["schema"] in ("operator-harness/artifact-index@1", "operator-harness/diagnostic-evidence@1"), "unsupported artifact index")
        key = "files" if data["schema"] == "operator-harness/artifact-index@1" else "artifacts"
        require(isinstance(data.get(key), list) and bool(data[key]), "empty diagnostic inventory/artifact index")
        for item in data[key]:
            freeze(store, item, dependencies)


def checked(store, ref, dependencies):
    freeze(store, ref, dependencies)
    return json.loads(store.local(ref["path"]).read_text())


def item_or_ref(store, item, dependencies):
    if isinstance(item, dict) and set(item) == {"path", "sha256"}:
        return checked(store, item, dependencies)
    require(isinstance(item, dict), "expected evidence object or reference")
    return item


def plan(store, ref):
    dependencies = {}
    data = checked(store, ref, dependencies)
    require(data.get("schema") == "operator-harness/evaluation-plan@1", "unsupported evaluation plan")
    require(data.get("project_id") == store.project_id, "evaluation plan project mismatch")
    require(data.get("performance_kind") in ("measured", "absolute", "model", "none"), "invalid performance kind")
    required = {"requirements", "reference", "comparator", "cases", "protocol", "environment"}
    require(isinstance(data.get("inputs"), dict) and set(data["inputs"]) == required,
            "evaluation plan needs frozen requirements/reference/comparator/cases/protocol/environment")
    for item in data["inputs"].values():
        freeze(store, item, dependencies)
    execution = data.get("execution")
    require(isinstance(execution, dict) and bool(execution), "evaluation plan needs a frozen execution adapter")
    require(execution.get("backend") in METRICS, "unsupported execution backend")
    require(execution.get("metric", METRICS[execution["backend"]]) == METRICS[execution["backend"]],
            "execution metric does not match backend")
    freeze(store, execution["adapter"], dependencies)
    require(number(execution.get("timeout_seconds"), positive=True), "invalid execution timeout")
    cases = data.get("cases")
    require(isinstance(cases, list) and bool(cases), "evaluation plan needs mandatory cases")
    ids = []
    for case in cases:
        require(isinstance(case.get("id"), str) and bool(case["id"]), "missing case ID")
        ids.append(case["id"])
        outputs = case.get("outputs")
        require(isinstance(outputs, list) and bool(outputs) and
                all(isinstance(x, str) and x for x in outputs) and len(set(outputs)) == len(outputs),
                "case needs distinct mandatory outputs")
        if data["performance_kind"] != "none":
            require(type(case.get("min_samples")) is int and case["min_samples"] >= 2,
                    "case needs at least two performance samples")
            require(number(case.get("max_latency_ms"), positive=True), "invalid frozen latency target")
    require(len(set(ids)) == len(ids), "duplicate mandatory case")
    require(data.get("statistic") in ("mean", "median"), "unsupported performance statistic")
    return data, dependencies


def comparison_identity(data):
    comparison = {key: data[key] for key in ("cases", "performance_kind", "statistic")}
    comparison["inputs"] = {key: ref["sha256"] for key, ref in data["inputs"].items()}
    backend = data["execution"]["backend"]
    comparison["measurement"] = {"backend": backend, "metric": METRICS[backend]}
    return hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()


def previous_comparison_identity(data):
    """Exact pre-metric identity, used only to recognize frozen historical records."""
    comparison = {key: data[key] for key in ("cases", "performance_kind", "statistic")}
    comparison["inputs"] = {key: ref["sha256"] for key, ref in data["inputs"].items()}
    return hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()


def matches_best_comparison(store, tree, comparison):
    best = tree["best"]
    if best["comparison"] == comparison:
        return True
    run = tree["runs"].get(best["run_id"], {})
    accepted = run.get("acceptance", {}).get("validation", {})
    if not (run.get("result") and run.get("evaluation_plan") and accepted.get("eligible") is True
            and accepted.get("candidate_id") == best["candidate_id"]
            and accepted.get("comparison") == best["comparison"]):
        return False
    # Check original evidence integrity and its frozen execution contract. This
    # does not run a new evaluator or change the historical acceptance decision.
    store.verify_run(tree, best["run_id"])
    data, _ = plan(store, run["evaluation_plan"])
    return (best["comparison"] == previous_comparison_identity(data)
            and comparison_identity(data) == comparison)


def device_receipt(store, ref, context, worker_result, dependencies):
    """Validate a final device observation; callers must obtain it from their guard."""
    receipt = checked(store, ref, dependencies)
    require(receipt.get("schema") == "operator-harness/device-receipt@1", "unsupported device receipt")
    for key in ("project_id", "run_id", "candidate_id", "request_id", "artifacts", "evaluation_plan"):
        require(receipt.get(key) == context[key], "device receipt identity mismatch: " + key)
    require(receipt.get("worker_result") == worker_result, "device receipt worker result mismatch")
    freeze(store, worker_result, dependencies)
    exits = receipt.get("exit_codes")
    require(receipt.get("status") == "released" and receipt.get("conflicts") == [],
            "device guard is pending or conflicted")
    require(isinstance(exits, dict) and "guard" in exits and
            all(isinstance(key, str) and key and type(value) is int and value == 0
                for key, value in exits.items()), "device guard/layer did not exit successfully")
    files = receipt.get("raw_files")
    require(isinstance(files, list) and bool(files), "device receipt needs raw guard observations")
    for item in files:
        freeze(store, item, dependencies)
    device_usage(store, context, receipt)
    return receipt


def device_usage(store, context, receipt):
    """A device reservation covers the guard lifetime, not only the local worker."""
    control = store.load().get("control")
    if control is None:
        return None
    job = control["jobs"].get(context["request_id"])
    require(job and job["run_id"] == context["run_id"] and job["status"] == "exited",
            "device finalization needs the matching exited process job")
    key = job["reservation_id"]
    reservation = control["reservations"][key]
    pool = control["budgets"][reservation["pool"]]
    usage = receipt.get("resource_usage")
    require(isinstance(usage, dict) and set(usage) == {"pool", "unit", "used"} and
            usage["pool"] == reservation["pool"] and usage["unit"] == pool["unit"] and
            number(usage["used"], positive=True), "device receipt needs actual matching resource usage")
    if pool["unit"] == "executions":
        require(type(usage["used"]) is int and usage["used"] == 1,
                "one started device request consumes one execution")
    elif pool["unit"] == "seconds":
        path = store.records / "runs" / context["run_id"] / "jobs" / context["request_id"] / "process-receipt.json"
        process = json.loads(path.read_text())
        require(all(process.get(key) == context[key] for key in ("project_id", "run_id", "request_id")) and
                process.get("status") == "exited" and process.get("exit_codes") == {"process": 0} and
                number(process.get("elapsed_seconds")), "device usage needs the original successful process receipt")
        require(usage["used"] >= process["elapsed_seconds"],
                "guard lifetime usage cannot be shorter than the observed process duration")
    else:
        require(False, "unsupported device resource unit")
    if reservation["settled"]:
        require(reservation["used"] == usage["used"], "device usage disagrees with prior settlement")
    return {"id": key, "used": usage["used"], "settled": reservation["settled"]}


def validation(store, run, result):
    """Bind output, raw case reports and receipts to the pre-registered evaluator.

    Raw accuracy reports remain claims from the frozen comparator; hashes do not
    authenticate execution or recompute arbitrary tensor comparisons.
    """
    scope = run["task"]
    data, dependencies = plan(store, run["evaluation_plan"])
    ref = store.reference(store.value(result, "validation_evidence"))
    report = checked(store, ref, dependencies)
    require(report.get("schema") == "operator-harness/validation@1", "unsupported validation evidence")
    run_id = store.value(scope, "run_id")
    require(report.get("project_id") == store.project_id and report.get("run_id") == run_id,
            "validation run/project mismatch")
    require(report.get("evaluation_plan") == run["evaluation_plan"], "validation plan mismatch")
    candidate = store.reference(store.value(result, "candidate_manifest"))
    require(report.get("candidate") == candidate, "validation candidate manifest mismatch")
    store.contract(candidate, "candidate", required=True)
    candidate_id = store.value(candidate, "candidate_id")
    store.markdown(candidate, candidate_id=candidate_id, search_node_id=run["search_node_id"])
    require(report.get("candidate_id") == candidate_id, "validation candidate identity mismatch")
    mode = store.value(scope, "operation")
    if mode in ("verify-only", "collect-only"):
        require(candidate_id == run["candidate_id"] and candidate == run.get("input_candidate"),
                "read-only validation changed candidate")
    if mode == "implement-and-validate":
        require(candidate_id != run["candidate_id"], "implementation must produce a new candidate")
        store.parent(candidate, run["candidate_id"], run.get("input_candidate"))
    dependencies[candidate["path"]] = candidate
    artifacts = report.get("artifacts")
    require(isinstance(artifacts, dict) and set(artifacts) == {"source", "build", "launch"},
            "validation needs source/build/launch artifacts")
    for item in artifacts.values():
        freeze(store, item, dependencies)
    for name, item in artifacts.items():
        require(store.value(candidate, name + "_sha256") == item["sha256"],
                "candidate artifact identity mismatch: " + name)
    receipts = report.get("receipts")
    require(isinstance(receipts, list) and bool(receipts), "validation needs job receipts")
    jobs = set()
    for item in receipts:
        receipt = item_or_ref(store, item, dependencies)
        require(receipt.get("schema") == "operator-harness/execution-receipt@1", "unsupported receipt")
        require(receipt.get("project_id") == store.project_id and receipt.get("run_id") == run_id
                and receipt.get("candidate_id") == candidate_id, "receipt identity mismatch")
        request = receipt.get("request_id")
        require(isinstance(request, str) and request and request not in jobs, "invalid/duplicate receipt request")
        jobs.add(request)
        exits = receipt.get("exit_codes")
        require(receipt.get("status") == "exited" and isinstance(exits, dict) and bool(exits)
                and all(type(x) is int and x == 0 for x in exits.values()), "job not successfully exited")
        require(receipt.get("valid") is True, "receipt not validated by execution adapter")
        require(receipt.get("producer") == "evaluate@1" and
                receipt.get("backend") == data["execution"]["backend"], "execution backend/producer mismatch")
        process = checked(store, receipt["process"], dependencies)
        require(process.get("schema") == "operator-harness/process-receipt@1" and
                process.get("project_id") == store.project_id and process.get("run_id") == run_id and
                process.get("request_id") == request and process.get("status") == "exited" and
                process.get("exit_codes") == receipt["exit_codes"], "process receipt mismatch")
        if data["execution"]["backend"] == "torch_npu":
            require(receipt.get("device_receipt") and receipt.get("worker_result"), "missing final device receipt")
            device_receipt(store, receipt["device_receipt"], receipt, receipt["worker_result"], dependencies)
            produced = checked(store, receipt["worker_result"], dependencies)
            require(produced.get("status") == "complete" and produced.get("cases") == report.get("cases")
                    and produced.get("diagnostics") == report.get("diagnostics"), "device worker evidence mismatch")
        require(receipt.get("artifacts") == artifacts and receipt.get("evaluation_plan") == run["evaluation_plan"],
                "receipt execution/evaluator identity mismatch")
    rows = report.get("cases")
    require(isinstance(rows, list) and len(rows) == len(data["cases"]), "mandatory case coverage mismatch")
    require(len({row["id"] for row in rows}) == len(rows), "duplicate result case")
    by_id = {row["id"]: row for row in rows}
    accuracy_passed = True
    performance_passed = True
    measured = data["performance_kind"] != "none"
    for case in data["cases"]:
        require(case["id"] in by_id, "missing mandatory case: " + case["id"])
        row = by_id[case["id"]]
        for kind in ("accuracy", "performance") if measured else ("accuracy",):
            raw = item_or_ref(store, row[kind], dependencies)
            require(raw.get("schema") == "operator-harness/" + kind + "@1", "unsupported raw case report")
            require(raw.get("project_id") == store.project_id and raw.get("run_id") == run_id
                    and raw.get("candidate_id") == candidate_id and raw.get("case_id") == case["id"],
                    "raw case identity mismatch")
            require(raw.get("request_id") in jobs, "raw case has no matching exited job")
            require(raw.get("artifacts") == artifacts and raw.get("evaluation_plan") == run["evaluation_plan"],
                    "raw case execution/evaluator identity mismatch")
            raw_files = raw.get("raw_files")
            require(isinstance(raw_files, list) and bool(raw_files), "raw case needs output/sample artifacts")
            for item in raw_files:
                freeze(store, item, dependencies)
            if kind == "accuracy":
                outputs = raw.get("outputs")
                require(isinstance(outputs, dict) and set(outputs) == set(case["outputs"])
                        and all(type(x) is bool for x in outputs.values()), "mandatory output coverage mismatch")
                accuracy_passed = accuracy_passed and all(outputs.values())
            else:
                require(raw.get("metric") == METRICS[data["execution"]["backend"]],
                        "performance metric does not match execution backend")
                samples = raw.get("samples_ms")
                require(isinstance(samples, list) and len(samples) >= case["min_samples"]
                        and all(number(x, positive=True) for x in samples), "invalid/insufficient performance samples")
                statistic = statistics.mean(samples) if data["statistic"] == "mean" else statistics.median(samples)
                performance_passed = performance_passed and statistic <= case["max_latency_ms"]
    accuracy = "passed" if accuracy_passed else "failed"
    performance = ("passed" if performance_passed else "failed") if measured else "not_applicable"
    require(store.value(result, "accuracy_status") == accuracy, "declared accuracy disagrees with case evidence")
    require(store.value(result, "performance_status") == performance, "declared performance disagrees with samples")
    if store.value(scope, "performance_required") == "true":
        require(measured, "required performance plan is absent")
    data_path = store.reference(store.value(result, "analysis_data"))
    store.markdown(data_path, candidate_id=candidate_id)
    # Frozen raw diagnostic inventory is explicit, not arbitrary Markdown recursion.
    diagnostics = report.get("diagnostics", [])
    require(isinstance(diagnostics, list), "invalid diagnostic inventory")
    if store.value(scope, "diagnostics_required") == "true":
        require(bool(diagnostics), "required diagnostic artifacts absent")
    for item in diagnostics:
        freeze(store, item, dependencies)
    return {"manifest": ref, "dependencies": dependencies, "candidate": candidate,
            "candidate_id": candidate_id, "requests": sorted(jobs),
            "comparison": comparison_identity(data),
            "eligible": accuracy_passed and (not measured or performance_passed)}


def limited_validation(store, run, result):
    """Code-only and diagnostic-only runs cannot earn candidate promotion."""
    candidate = store.reference(store.value(result, "candidate_manifest"))
    store.contract(candidate, "candidate", required=True)
    candidate_id = store.value(candidate, "candidate_id")
    store.markdown(candidate, search_node_id=run["search_node_id"])
    store.markdown(store.reference(store.value(result, "analysis_data")), candidate_id=candidate_id)
    dependencies = {candidate["path"]: candidate}
    require(store.value(result, "accuracy_status") in ("not_run", "not_applicable") and
            store.value(result, "performance_status") in ("not_run", "not_applicable"),
            "limited mode links prior validation; it cannot declare new accuracy/performance acceptance")
    requests, manifest = [], None
    if store.value(run["task"], "operation") == "collect-only":
        require(candidate_id == run["candidate_id"] and candidate == run.get("input_candidate"),
                "collect-only changed candidate")
        ref = store.reference(store.value(result, "diagnostic_evidence"))
        report = checked(store, ref, dependencies)
        require(report.get("schema") == "operator-harness/diagnostic-evidence@1"
                and report.get("project_id") == store.project_id
                and report.get("run_id") == store.value(run["task"], "run_id")
                and report.get("candidate") == candidate, "diagnostic evidence identity mismatch")
        require(isinstance(report.get("artifacts"), list) and bool(report["artifacts"]),
                "empty diagnostic inventory")
        for item in report["artifacts"]:
            freeze(store, item, dependencies)
        if run.get("evaluation_plan"):
            data, plan_dependencies = plan(store, run["evaluation_plan"])
            dependencies.update(plan_dependencies)
            if data["execution"]["backend"] == "torch_npu":
                require(report.get("device_receipt") and report.get("worker_result"), "missing final device receipt")
                context = dict(report, candidate_id=candidate_id, evaluation_plan=run["evaluation_plan"],
                               artifacts={key: store.reference(store.value(candidate, key))
                                          for key in ("source", "build", "launch")})
                device_receipt(store, report["device_receipt"], context, report["worker_result"], dependencies)
                produced = checked(store, report["worker_result"], dependencies)
                require(produced.get("status") == "complete" and produced.get("diagnostics") == report["artifacts"],
                        "device worker evidence mismatch")
                process = checked(store, report["process"], dependencies)
                require(process.get("schema") == "operator-harness/process-receipt@1" and
                        process.get("project_id") == store.project_id and process.get("run_id") == context["run_id"] and
                        process.get("request_id") == context["request_id"] and process.get("status") == "exited" and
                        process.get("exit_codes") == {"process": 0}, "process receipt mismatch")
        requests = [report["request_id"]] if report.get("request_id") else []
        manifest = ref
    return {"dependencies": dependencies, "candidate": candidate, "candidate_id": candidate_id,
            "manifest": manifest, "requests": requests, "eligible": False}
