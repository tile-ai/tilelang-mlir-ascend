"""Cooperative control ledger inside search/tree.json, under the same transaction."""
import json
from datetime import datetime

from evidence import number, require

TERMINAL = {"exited", "cancelled", "not_submitted"}
TRANSITIONS = {
    "intent": {"submitted", "running", "unknown", "exited", "not_submitted"},
    "submitted": {"running", "unknown", "exited", "cancelled"},
    "running": {"unknown", "exited", "cancelled"},
    "unknown": {"submitted", "running", "exited", "cancelled", "not_submitted"},
}


def available(control, pool):
    budget = control["budgets"][pool]
    consumed = sum(r["used"] if r["settled"] else r["amount"]
                   for r in control["reservations"].values() if r["pool"] == pool)
    return budget["limit"] - consumed


def reconciled(tree, run_id):
    c = tree.get("control", {})
    require(all(j["status"] in TERMINAL for j in c.get("jobs", {}).values() if j["run_id"] == run_id),
            "run has outstanding/unknown jobs")
    require(all(r["settled"] for r in c.get("reservations", {}).values() if r["run_id"] == run_id),
            "run has unsettled resources")


def admission(tree, operation):
    c = tree.get("control")
    if c is None or operation["op"] not in ("run", "develop-batch"):
        return
    runs = ([operation["id"]] if operation["op"] == "run" else
            [member["id"] for member in operation["members"]])
    require(not c["paused"], "control is paused")
    require(len(c["steps"]) + len(runs) <= c["step_limit"], "control step budget exhausted")
    c["steps"].extend(runs)


def observation(store, ref, run_id, request_id, status):
    store.verify_ref(ref)
    data = json.loads(store.local(ref["path"]).read_text())
    require(data.get("schema") == "operator-harness/job-observation@1" and
            data.get("project_id") == store.project_id and data.get("run_id") == run_id and
            data.get("request_id") == request_id and data.get("status") == status,
            "job observation identity/state mismatch")
    observed = datetime.fromisoformat(data["observed_at"])
    require(observed.tzinfo is not None, "job observation needs timezone")
    require(isinstance(data.get("source"), str) and data["source"], "job observation needs query source")
    return observed


def apply(store, tree, op):
    action = op["op"]
    if action == "control-init":
        require("control" not in tree, "control already initialized; cannot reset steps")
        require(not any(not r.get("result") and not r.get("closure") for r in tree["runs"].values()),
                "initialize control at a safe boundary")
        require(type(op.get("step_limit")) is int and op["step_limit"] > 0, "invalid step limit")
        budgets = op.get("budgets")
        require(isinstance(budgets, dict), "budgets must be an object")
        for key, budget in budgets.items():
            require(isinstance(key, str) and key and number(budget.get("limit"))
                    and isinstance(budget.get("unit"), str) and budget["unit"], "invalid budget")
        tree["control"] = {"step_limit": op["step_limit"], "steps": [], "budgets": budgets,
                           "reservations": {}, "jobs": {}, "paused": False,
                           "authority": store.reference(op["authority"])}
        return
    require("control" in tree, "initialize control first")
    c = tree["control"]
    if action in ("control-pause", "control-resume", "control-extend"):
        c["authority"] = store.reference(op["authority"])
        if action == "control-extend":
            limit = op["step_limit"]
            require(type(limit) is int and limit >= c["step_limit"], "extension cannot lower step limit")
            require(set(op["limits"]) == set(c["budgets"]), "extension must preserve budget pools")
            for key, value in op["limits"].items():
                require(number(value) and value >= c["budgets"][key]["limit"], "invalid budget extension")
                c["budgets"][key]["limit"] = value
            c["step_limit"] = limit
        else:
            c["paused"] = action == "control-pause"
    elif action == "control-reserve":
        require(not c["paused"], "control is paused")
        run = tree["runs"][op["run_id"]]
        require(not run.get("result") and not run.get("closure"), "run already settled")
        key, pool = op["id"], op["pool"]
        require(isinstance(key, str) and key and key not in c["reservations"], "reservation already exists/invalid")
        require(pool in c["budgets"] and number(op["amount"], positive=True), "invalid reservation")
        require(op["amount"] <= available(c, pool), "resource budget exhausted")
        c["reservations"][key] = {"run_id": op["run_id"], "pool": pool, "amount": op["amount"],
                                  "settled": False, "used": 0}
    elif action == "control-job":
        key = op["id"]
        require(isinstance(key, str) and key, "invalid request ID")
        status = op["status"]
        old = c["jobs"].get(key)
        if old is None:
            require(not c["paused"] and status == "intent", "new job requires unpaused intent")
            reservation = c["reservations"][op["reservation_id"]]
            require(not reservation["settled"], "reservation already settled")
            observation(store, store.reference(op["evidence"]), reservation["run_id"], key, status)
            c["jobs"][key] = {"run_id": reservation["run_id"], "reservation_id": op["reservation_id"],
                               "status": "intent", "evidence": store.reference(op["evidence"])}
        else:
            require(status in TRANSITIONS.get(old["status"], set()), "invalid job transition")
            ref = store.reference(op["evidence"])
            observed = observation(store, ref, old["run_id"], key, status)
            previous = observation(store, old["evidence"], old["run_id"], key, old["status"])
            require(observed >= previous, "stale job observation")
            old.update(status=status, evidence=ref)
    elif action == "control-settle":
        reservation = c["reservations"][op["id"]]
        require(not reservation["settled"] and number(op["used"]), "reservation already settled/invalid usage")
        require(all(j["status"] in TERMINAL for j in c["jobs"].values()
                    if j["reservation_id"] == op["id"]), "unknown/outstanding job keeps reservation")
        # Record real overruns instead of hiding them; future reservations see the deficit.
        reservation.update(settled=True, used=op["used"], evidence=store.reference(op["evidence"]))
    else:
        raise ValueError("unknown control operation: " + action)


def validate(store, tree, artifacts=True):
    c = tree.get("control")
    if c is None:
        return
    if artifacts:
        store.verify_ref(c["authority"])
    require(type(c["paused"]) is bool and type(c["step_limit"]) is int
            and c["step_limit"] >= len(c["steps"]), "invalid control state")
    require(len(c["steps"]) == len(set(c["steps"])) and all(r in tree["runs"] for r in c["steps"]),
            "invalid control steps")
    for reservation in c["reservations"].values():
        require(reservation["run_id"] in tree["runs"] and reservation["pool"] in c["budgets"]
                and number(reservation["amount"], positive=True) and number(reservation["used"])
                and type(reservation["settled"]) is bool, "invalid reservation state")
        if artifacts and reservation["settled"]:
            store.verify_ref(reservation["evidence"])
    for request_id, job in c["jobs"].items():
        require(job["reservation_id"] in c["reservations"] and
                c["reservations"][job["reservation_id"]]["run_id"] == job["run_id"] and
                job["status"] in set(TRANSITIONS) | TERMINAL, "invalid job state")
        require(not c["reservations"][job["reservation_id"]]["settled"] or job["status"] in TERMINAL,
                "unresolved job has released reservation")
        if artifacts:
            observation(store, job["evidence"], job["run_id"], request_id, job["status"])
    for key, run in tree["runs"].items():
        if run.get("result") or run.get("closure"):
            reconciled(tree, key)


def snapshot(tree):
    c = tree.get("control")
    if c is None:
        return {"enabled": False}
    return {"enabled": True, "paused": c["paused"], "steps_used": len(c["steps"]),
            "step_limit": c["step_limit"],
            "available": {key: available(c, key) for key in c["budgets"]},
            "outstanding_jobs": {key: job for key, job in c["jobs"].items() if job["status"] not in TERMINAL},
            "unsettled_reservations": [key for key, r in c["reservations"].items() if not r["settled"]]}


def receipts(store, tree, run_id, validation):
    if "control" not in tree or not validation:
        return
    requests = validation.get("requests", [])
    if not requests and not validation.get("manifest"):
        return  # Code-only contains no execution evidence.
    require(bool(requests), "execution evidence needs registered jobs")
    for request in requests:
        job = tree["control"]["jobs"].get(request)
        require(job is not None and job["run_id"] == run_id and job["status"] == "exited",
                "execution receipt has no matching exited control job")
        observation(store, job["evidence"], run_id, request, "exited")
