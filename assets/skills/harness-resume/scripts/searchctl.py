#!/usr/bin/env python3
"""Project search and knowledge records. Python 3.9+, POSIX, stdlib only.

This is a metadata ledger, not an executor or an authentication service.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness-new-project/scripts"))
import projectctl
from records import metadata
import evidence
import control
from evidence import Invalid



def require(condition, message):
    if not condition:
        raise Invalid(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def fields(record, *names):
    require(isinstance(record, dict), "expected an object")
    for name in names:
        require(nonempty(record.get(name)), f"missing/non-string {name}")


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value),
            "invalid identifier")
    return value


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def settled(run):
    return bool(run.get("result") or run.get("closure"))


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value, immutable=False):
    data = value if isinstance(value, bytes) else encoded(value)
    if immutable and path.exists():
        require(path.read_bytes() == data, f"immutable record conflict: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


class Store:
    def __init__(self, project, project_id=None, owner=None, legacy=False):
        self.root = Path(project).resolve()
        require(self.root.is_dir(), "project directory must exist")
        self.workspace = projectctl.Workspace(self.root)
        self.legacy = legacy
        self.owner = owner
        require(not (legacy and project_id), "legacy and named project are mutually exclusive")
        self.project_id = None if legacy else self.workspace.resolve(project_id)
        self.records = self.root / ".workspace" if legacy else self.workspace.project(self.project_id)
        self.directory = (self.records / "search" if legacy else
                          self.workspace.local(self.project_id, "search"))
        self.path = self.directory / "tree.json"

    @contextmanager
    def write(self, closing=False):
        require(not self.legacy, "legacy search is read-only; explicitly migrate before writing")
        states = ("active", "freezing") if closing else ("active",)
        with self.workspace.edit(self.project_id, self.owner, states):
            with lock(self.directory):
                yield

    def local(self, relative):
        require(nonempty(relative) and not Path(relative).is_absolute(), "expected project-relative path")
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "path escapes project")
        if not self.legacy and path.is_relative_to(self.root / ".workspace"):
            require(path.is_relative_to(self.records) or
                    path.is_relative_to(self.root / ".workspace/environments"),
                    "reference belongs to another project")
        return path

    def reference(self, path):
        file = self.local(path)
        return {"path": path, "sha256": digest(file.read_bytes())}

    def snapshot(self, source):
        data = self.local(source).read_bytes()
        path = str((self.records / "knowledge/evidence" / f"{digest(data)}.snapshot").relative_to(self.root))
        atomic(self.local(path), data, immutable=True)
        return self.reference(path)

    def verify_ref(self, ref):
        require(isinstance(ref, dict), "invalid artifact reference")
        require(self.reference(ref["path"]) == ref, f"artifact changed: {ref['path']}")

    def load(self):
        tree = read(self.path)
        self.validate(tree)
        return tree

    def validate(self, tree):
        """Validate ledger topology without reading historical artifacts."""
        require(isinstance(tree, dict) and tree.get("schema") == "operator-harness/search@1", "unsupported search schema")
        require(self.legacy or tree.get("project_id") == self.project_id, "search project identity mismatch")
        fields(tree, "operator", "task_contract")
        nodes, candidates, runs = tree["nodes"], tree["candidates"], tree["runs"]
        require(tree["current_node"] in nodes, "unknown current node")
        require(sum(n["parent"] is None for n in nodes.values()) == 1, "tree must have one root")
        for collection, parent_key in ((nodes, "parent"), (candidates, "parent_candidate")):
            for key, item in collection.items():
                identifier(key)
                cursor, seen = key, set()
                while cursor is not None:
                    require(cursor in collection and cursor not in seen, "invalid lineage or cycle")
                    seen.add(cursor)
                    cursor = collection[cursor][parent_key]
        for node in nodes.values():
            fields(node, "hypothesis", "scope")
            require(node["status"] in ("open", "parked", "closed"), "invalid node status")
        for candidate in candidates.values():
            require(candidate["search_node_id"] in nodes, "unknown candidate node")
        for key, run in runs.items():
            identifier(key)
            require(run["search_node_id"] in nodes, "unknown run node")
            require(run["stage"] in ("prepare-environment", "establish-baseline", "port-baseline", "analysis", "develop", "device-test"), "invalid stage")
            require(run["candidate_id"] is None or run["candidate_id"] in candidates, "unknown input candidate")
            require(not (run.get("result") and run.get("closure")), "closed run cannot also be accepted")
            if run.get("batch_id"):
                require(key in tree["batches"][run["batch_id"]]["members"], "unlisted batch run")
        for key, batch in tree.get("batches", {}).items():
            identifier(key)
            require(len(batch["members"]) >= 2 and len(set(batch["members"])) == len(batch["members"]), "invalid batch members")
            require(set(batch["inputs"]) == {"requirements", "baseline", "cases", "protocol", "environment"}, "batch needs frozen comparison inputs")
            repos = set()
            for member in batch["members"]:
                run = runs[member]
                require(run.get("batch_id") == key and run["stage"] == "develop", "batch run mismatch")
                require(run["candidate_id"] == batch["candidate_id"], "batch candidate mismatch")
                analysis = runs[run["analysis_run"]]
                require(analysis["stage"] == "analysis" and analysis.get("result") and
                        analysis["candidate_id"] == run["candidate_id"] and analysis["search_node_id"] == run["search_node_id"], "inapplicable analysis run")
                require(run["repo_ids"] and len(set(run["repo_ids"])) == len(run["repo_ids"]) and not repos.intersection(run["repo_ids"]), "batch worktrees overlap")
                repos.update(run["repo_ids"])
        for key, revisions in tree["hot"].items():
            identifier(key)
            require(bool(revisions), "empty hot history")
        for binding in tree["cooled_inputs"]:
            require(binding["search_node_id"] in nodes, "unknown knowledge consumer")
            fields(binding, "reason")
        control.validate(self, tree, artifacts=False)

    def audit(self, tree):
        """Explicit historical integrity scan. Damage is reported, not a load lock."""
        issues, seen = [], set()
        def visit(value, context):
            if isinstance(value, dict):
                if set(value) == {"path", "sha256"}:
                    key = (value["path"], value["sha256"])
                    if key in seen:
                        return
                    seen.add(key)
                    try:
                        self.verify_ref(value)
                    except (ValueError, OSError, KeyError) as exc:
                        issues.append({"context": context, "path": value["path"], "error": str(exc)})
                    return
                for key, item in value.items():
                    if key != "events":
                        visit(item, context + "." + key)
            elif isinstance(value, list):
                for item in value:
                    visit(item, context)
        visit(tree, "search")
        for key, revisions in tree.get("hot", {}).items():
            for ref in revisions:
                try:
                    hot = read(self.local(ref["path"]))
                    visit(hot, "search.hot." + key)
                except (ValueError, OSError) as exc:
                    issues.append({"context": "search.hot." + key,
                                   "path": ref["path"], "error": str(exc)})
        return issues

    def parent(self, manifest, parent_id, parent_ref=None):
        actual = self.value(manifest, "parent_candidate")
        if parent_id is None:
            actual = re.sub(r"^not_applicable\s*[:：]\s*.+$", "not_applicable", actual)
        # Historical manifests used paths; newly emitted metadata uses IDs.
        allowed = {parent_id} if parent_id else {"not_applicable"}
        if parent_ref and '<!-- harness-metadata -->' not in self.local(manifest["path"]).read_text():
            allowed.add(parent_ref["path"])
        require(actual in allowed, "candidate parent mismatch")

    def verify_run(self, tree, run_id):
        run = tree["runs"][run_id]
        self.verify_ref(run["task"])
        if run.get("input_candidate"):
            self.verify_ref(run["input_candidate"])
        if run.get("evaluation_plan"):
            self.verify_ref(run["evaluation_plan"])
            for ref in run.get("evaluation_inputs", {}).values():
                self.verify_ref(ref)
        if run.get("batch_id"):
            for ref in tree["batches"][run["batch_id"]]["inputs"].values():
                self.verify_ref(ref)
        if run.get("result"):
            self.verify_ref(run["result"])
            for issue in self.audit(run.get("acceptance", {})):
                raise Invalid("accepted evidence changed: " + issue["path"])

    def validate_hot(self, hot, tree):
        fields(hot, "statement", "applicability", "limitations")
        require(hot["kind"] in ("observation", "hypothesis", "conclusion"), "invalid knowledge kind")
        require(isinstance(hot["nodes"], list) and bool(hot["nodes"]), "hot needs source nodes")
        require(len(set(hot["nodes"])) == len(hot["nodes"]) and
                all(node in tree["nodes"] for node in hot["nodes"]), "invalid hot nodes")
        require(isinstance(hot["evidence"], list) and bool(hot["evidence"]), "hot needs evidence")
        for ref in hot["evidence"]:
            self.verify_ref(ref)

    def init(self, operator, task_contract, hypothesis, scope):
        with self.write():
            require(not self.path.exists(), "tree already initialized")
            tree = {"schema": "operator-harness/search@1", "project_id": self.project_id, "operator": operator,
                    "task_contract": task_contract, "current_node": "N0",
                    "nodes": {"N0": {"parent": None, "hypothesis": hypothesis,
                                      "scope": scope, "status": "open"}},
                    "candidates": {}, "runs": {}, "hot": {}, "cooled_inputs": [],
                    "events": []}
            self.validate(tree)
            atomic(self.path, tree)
            return tree

    def apply(self, operation):
        """One idempotent, serialized operation; immutable blobs precede its index."""
        with self.write(closing=operation.get("op") in ("accept", "close-run", "control-job", "control-settle", "control-pause")):
            tree = self.load()
            event_id = identifier(operation["event_id"])
            for event in tree["events"]:
                if event["operation"]["event_id"] == event_id:
                    require(event["operation"] == operation, "event ID reused with different content")
                    return tree
            self.mutate(tree, operation)
            control.admission(tree, operation)
            self.validate(tree)
            tree["events"].append({"at": now(), "operation": deepcopy(operation)})
            atomic(self.path, tree)
            return tree

    def mutate(self, tree, op):
        action = op["op"]
        if action.startswith("control-"):
            control.apply(self, tree, op)
        elif action == "add-node":
            key = identifier(op["id"])
            require(key not in tree["nodes"], "node already exists")
            require(op["parent"] in tree["nodes"], "unknown parent")
            fields(op, "hypothesis", "scope")
            tree["nodes"][key] = {name: op[name] for name in ("parent", "hypothesis", "scope")}
            tree["nodes"][key]["status"] = "open"
            if op.get("plan"):
                ref = self.reference(op["plan"])
                plan = self.search_plan(ref)
                prior = [k for k, node in tree["nodes"].items() if node.get("mechanism_id") == plan["mechanism_id"]]
                require(set(prior) <= set(plan["prior_nodes"]), "search plan omits matching prior mechanisms")
                require(all(k in tree["nodes"] and k != key for k in plan["prior_nodes"]), "unknown prior search node")
                require(not prior or plan["new_evidence"] or nonempty(plan.get("changed_conditions")),
                        "repeated mechanism needs new evidence or changed conditions")
                tree["nodes"][key].update(plan=ref, mechanism_id=plan["mechanism_id"])
        elif action in ("select", "node-status"):
            require(op["id"] in tree["nodes"], "unknown node")
            fields(op, "reason")
            if action == "select":
                require(tree["nodes"][op["id"]]["status"] == "open", "reopen node before selecting")
                require(all(settled(run) for run in tree["runs"].values()),
                        "accept outstanding runs before switching strategy")
                tree["current_node"] = op["id"]
            else:
                require(op["status"] in ("open", "parked", "closed"), "invalid status")
                require(op["status"] == "open" or all(
                    settled(run) or run["search_node_id"] != op["id"]
                    for run in tree["runs"].values()), "node has outstanding runs")
                tree["nodes"][op["id"]]["status"] = op["status"]
        elif action == "candidate":
            key = identifier(op["id"])
            require(key not in tree["candidates"], "candidate already registered")
            require(op["search_node_id"] in tree["nodes"], "unknown node")
            ref = self.reference(op["manifest"])
            self.markdown(ref, candidate_id=key, search_node_id=op["search_node_id"])
            parent = op["parent_candidate"]
            if parent is not None:
                require(parent in tree["candidates"], "unknown parent candidate")
            self.contract(ref, "candidate")
            self.parent(ref, parent, tree["candidates"][parent]["manifest"] if parent else None)
            tree["candidates"][key] = {"search_node_id": op["search_node_id"],
                                       "parent_candidate": parent, "manifest": ref}
        elif action == "run":
            require(op["stage"] in ("prepare-environment", "establish-baseline", "port-baseline", "analysis", "develop"),
                    "new runs use the five current stages; device-test is legacy only")
            require(not any(run.get("batch_id") and not settled(run)
                            for run in tree["runs"].values()), "develop batch is outstanding")
            require(not any(not settled(run) and
                            (op["stage"] != "analysis" or run["stage"] != "analysis")
                            for run in tree["runs"].values()),
                    "ordinary develop/port-baseline must run alone; use develop-batch for parallel optimization")
            key = identifier(op["id"])
            require(key not in tree["runs"], "run already registered")
            require(op["search_node_id"] == tree["current_node"], "run must use selected node")
            require(tree["nodes"][tree["current_node"]]["status"] == "open", "node is not open")
            ref = self.reference(op["task"])
            self.markdown(ref, run_id=key, node=op["stage"], search_node_id=op["search_node_id"])
            if op["stage"] == "develop":
                self.develop_scope(ref)
            elif op["stage"] == "port-baseline":
                self.port_scope(ref)
            tree["runs"][key] = {"search_node_id": op["search_node_id"], "stage": op["stage"],
                                  "candidate_id": op["candidate_id"], "task": ref, "result": None}
            self.admit(tree["runs"][key], op.get("legacy", False), tree)
            if op["stage"] == "develop":
                tree["runs"][key].setdefault("acceptance_policy", "develop-acceptance@1")
            elif op["stage"] == "port-baseline":
                tree["runs"][key].setdefault("acceptance_policy", "port-acceptance@1")
        elif action == "develop-batch":
            self.develop_batch(tree, op)
        elif action == "accept":
            run = tree["runs"][op["id"]]
            control.reconciled(tree, op["id"])
            self.verify_run(tree, op["id"])
            require(not settled(run), "run already accepted or closed")
            ref = self.reference(op["result"])
            self.markdown(ref, run_id=op["id"], node=run["stage"], search_node_id=run["search_node_id"])
            self.contract(ref, "node-result", required=not run.get("legacy", "evaluator" not in run))
            if "evaluator" in run:
                require(run["evaluator"] == evidence.evaluator_identity(),
                        "evaluator changed; use the recorded tool bundle or supersede the task")
            self.terminal(ref)
            if run["stage"] == "develop":
                if op.get("scope"):
                    require("acceptance_policy" not in run, "new develop cannot replace its frozen scope")
                    run["acceptance_scope"] = self.reference(op["scope"])
                    self.markdown(run["acceptance_scope"], run_id=op["id"])
                run["acceptance"] = self.run_acceptance(run, ref)
            elif run["stage"] == "port-baseline":
                require(not op.get("scope"), "port-baseline cannot replace its frozen scope")
                run["acceptance"] = self.run_acceptance(run, ref)
            if run.get("batch_id"):
                self.markdown(ref, batch_id=run["batch_id"], analysis_run=run["analysis_run"])
            verified = run.get("acceptance", {}).get("validation", {})
            if verified:
                candidate = tree["candidates"].get(verified["candidate_id"])
                require(candidate and candidate["manifest"] == verified["candidate"], "register the exact output candidate before acceptance")
            control.receipts(self, tree, op["id"], verified)
            run["result"] = ref
        elif action == "close-run":
            run = tree["runs"][op["id"]]
            control.reconciled(tree, op["id"])
            require(not settled(run), "run already accepted or closed")
            require(op.get("disposition") in ("rejected", "abandoned", "superseded"), "invalid closure disposition")
            fields(op, "reason", "checkpoint", "evidence")
            successor = op.get("successor")
            if op["disposition"] == "superseded":
                identifier(successor)
                require(successor != op["id"] and successor not in tree["runs"], "successor must be a new run")
            else:
                require(successor is None, "only superseded runs name a successor")
            run["closure"] = {"disposition": op["disposition"], "reason": op["reason"],
                              "checkpoint": self.reference(op["checkpoint"]),
                              "evidence": self.reference(op["evidence"]), "successor": successor}
            self.validate_closure(op["id"], run)
        elif action == "promote-candidate":
            run = tree["runs"][op["run_id"]]
            self.verify_run(tree, op["run_id"])
            verified = run.get("acceptance", {}).get("validation", {})
            require(run.get("result") and verified.get("eligible") is True,
                    "promotion requires structured passing evidence")
            candidate_id = verified["candidate_id"]
            require(candidate_id in tree["candidates"] and
                    tree["candidates"][candidate_id]["manifest"] == verified["candidate"],
                    "register the exact validated candidate before promotion")
            fields(op, "reason")
            if tree.get("best"):
                require(tree["best"]["comparison"] == verified["comparison"],
                        "best candidate comparison changed; explicitly start a new comparison")
            tree["best"] = {"candidate_id": candidate_id, "run_id": op["run_id"],
                            "comparison": verified["comparison"], "reason": op["reason"]}
        elif action == "hot":
            key = identifier(op["id"])
            revisions = tree["hot"].setdefault(key, [])
            require(type(op["base_revision"]) is int and op["base_revision"] == len(revisions),
                    "stale hot revision")
            hot = {name: op[name] for name in ("id", "kind", "statement", "applicability", "limitations", "nodes")}
            hot["revision"] = len(revisions) + 1
            hot["evidence"] = [self.snapshot(path) for path in op["evidence"]]
            hot["evidence_sources"] = op["evidence"]
            self.validate_hot(hot, tree)
            # Content addressing allows safe retry after a crash before index commit.
            path = self.workspace.local(self.project_id, f"knowledge/hot/{key}/{digest(encoded(hot))}.json")
            atomic(path, hot, immutable=True)
            revisions.append(self.reference(str(path.relative_to(self.root))))
        elif action == "use-cooled":
            fields(op, "reason")
            ref = self.reference(op["artifact"])
            cooled = read(self.local(ref["path"]))
            validate_cooled(cooled)
            require(cooled["status"] == "active", "cooled knowledge needs revalidation or is superseded")
            self.latest_cooled(ref)
            tree["cooled_inputs"].append({"search_node_id": op["search_node_id"],
                                           "artifact": ref, "reason": op["reason"]})
        else:
            raise Invalid(f"unknown operation: {action}")

    def develop_batch(self, tree, op):
        """Admit an isolated group atomically; the host coordinates execution."""
        key = identifier(op["id"])
        require(key not in tree.get("batches", {}), "batch already registered")
        require(all(settled(run) for run in tree["runs"].values()),
                "accept outstanding runs before starting a batch")
        members = op["members"]
        require(isinstance(members, list) and len(members) >= 2, "batch needs at least two members")
        require(isinstance(op["inputs"], dict), "batch inputs must be an object")
        inputs = {name: self.reference(path) for name, path in op["inputs"].items()}
        data = self.workspace.load(self.project_id)
        paths = []
        ids = []
        comparison = None
        for member in members:
            run_id = identifier(member["id"])
            require(run_id not in tree["runs"], "run already registered")
            node = member["search_node_id"]
            require(node in tree["nodes"] and tree["nodes"][node]["status"] == "open", "node is not open")
            repo_ids = member["repo_ids"]
            require(isinstance(repo_ids, list) and bool(repo_ids), "member needs writable worktrees")
            for repo_id in repo_ids:
                identifier(repo_id)
                require(repo_id in data["repositories"], "unregistered worktree")
                repo = data["repositories"][repo_id]
                self.workspace.inventory(self.project_id, {"repositories": {repo_id: repo}})
                path = self.workspace.local(self.project_id, repo["path"]).resolve()
                require(not any(path.is_relative_to(other) or other.is_relative_to(path)
                                for other in paths), "batch worktrees overlap")
                paths.append(path)
            task = self.reference(member["task"])
            expected = self.records / "runs" / run_id / "TASK.md"
            require(self.local(member["task"]) == expected, "batch TASK must use its own run directory")
            self.markdown(task, run_id=run_id, node="develop", search_node_id=node,
                          batch_id=key, analysis_run=member["analysis_run"])
            self.develop_scope(task)
            require(member["analysis_run"] in tree["runs"], "unknown analysis run")
            analysis = tree["runs"][member["analysis_run"]]
            require(analysis.get("result"), "inapplicable analysis run")
            self.verify_ref(analysis["result"])
            self.markdown(analysis["result"], status="completed")
            tree["runs"][run_id] = {"search_node_id": node, "stage": "develop",
                                    "candidate_id": op["candidate_id"], "task": task, "result": None,
                                    "batch_id": key, "analysis_run": member["analysis_run"],
                                    "acceptance_policy": "develop-acceptance@1",
                                    "repo_ids": list(repo_ids)}
            self.admit(tree["runs"][run_id], op.get("legacy", False), tree)
            if not op.get("legacy", False):
                run = tree["runs"][run_id]
                require("evaluation_plan" in run, "batch members need full evaluation plans")
                plan, dependencies = evidence.plan(self, run["evaluation_plan"])
                for name in ("requirements", "cases", "protocol", "environment"):
                    require(plan["inputs"][name] == inputs.get(name), "batch evaluation inputs mismatch")
                require(inputs.get("baseline") in dependencies.values(), "batch baseline is absent from evaluation dependencies")
                identity = evidence.comparison_identity(plan)
                require(comparison is None or comparison == identity, "batch comparison plans differ")
                comparison = identity
            ids.append(run_id)
        tree.setdefault("batches", {})[key] = {"members": ids, "inputs": inputs,
                                               "candidate_id": op["candidate_id"]}

    def search_plan(self, ref):
        self.verify_ref(ref)
        plan = read(self.local(ref["path"]))
        require(plan.get("schema") == "operator-harness/search-plan@1" and
                plan.get("project_id") == self.project_id, "search plan schema/project mismatch")
        identifier(plan.get("mechanism_id"))
        fields(plan, "expected", "falsifier", "keep", "discard", "simplification")
        require(isinstance(plan.get("prior_nodes"), list) and isinstance(plan.get("new_evidence"), list),
                "search plan needs history and evidence lists")
        for item in plan["new_evidence"]:
            self.verify_ref(item)
        return plan

    def mechanisms(self):
        tree = self.load()
        return [{"node": key, "mechanism_id": node.get("mechanism_id"), "status": node["status"],
                 "hypothesis": node["hypothesis"], "plan": node.get("plan")}
                for key, node in tree["nodes"].items()]

    def contract(self, ref, kind, required=False):
        text = self.local(ref["path"]).read_text(encoding="utf-8")
        values = [metadata(text)["contract"]] if "contract" in metadata(text) else []
        versions = {"node-task": (1, 2, 3, 4), "node-result": (1, 2, 3, 4), "candidate": (1, 2)}
        if not values and not required:
            return None
        require(len(values) == 1 and values[0] in
                [f"operator-harness/{kind}@{v}" for v in versions[kind]], "unsupported/duplicate " + kind + " contract")
        if required:
            latest = "@2" if kind == "candidate" else "@4"
            require(values[0].endswith(latest), "new runs require current contract " + latest + "; legacy import must be explicit")
        return values[0]

    def admit(self, run, legacy, tree):
        require(type(legacy) is bool, "legacy must be a boolean")
        version = self.contract(run["task"], "node-task", required=not legacy)
        require(not legacy or version != "operator-harness/node-task@4", "@4 cannot downgrade to legacy")
        run["input_candidate"] = tree["candidates"].get(run["candidate_id"], {}).get("manifest")
        run["legacy"] = legacy
        run["evaluator"] = evidence.evaluator_identity()
        # Bundle provenance is informational; only evaluator code locks acceptance.
        run["harness_bundle"] = evidence.bundle_identity()
        if run["input_candidate"]:
            self.verify_ref(run["input_candidate"])
        if not legacy and run["stage"] == "develop":
            mode = self.value(run["task"], "operation")
            require(mode not in ("implement-and-validate", "verify-only", "collect-only") or run["candidate_id"] is not None,
                    "operation requires an input candidate")
        if legacy:
            require(self.value(run["task"], "operation") != "initial-and-validate"
                    if run["stage"] == "develop" else True, "initial mode requires @4")
        if legacy or run["stage"] not in ("develop", "port-baseline"):
            return
        run["acceptance_policy"] = ("develop" if run["stage"] == "develop" else "port") + "-acceptance@2"
        mode = self.value(run["task"], "operation")
        self.value(run["task"], "diagnostics_required")
        if mode != "code-only" and (mode != "collect-only" or
                "evaluation_plan" in metadata(self.local(run["task"]["path"]).read_text())):
            ref = self.reference(self.value(run["task"], "evaluation_plan"))
            data, dependencies = evidence.plan(self, ref)
            require(data.get("run_id") == self.value(run["task"], "run_id"), "evaluation plan run mismatch")
            if self.value(run["task"], "performance_required") == "true":
                require(data["performance_kind"] != "none", "required performance plan absent")
            if mode == "initial-and-validate":
                require(run["candidate_id"] is None and data["performance_kind"] in ("none", "absolute", "model"),
                        "initial mode requires no candidate and a non-measured performance goal")
                design = self.reference(self.value(run["task"], "initial_design"))
                require(bool(self.local(design["path"]).read_bytes().strip()), "empty initial design")
                run["initial_design"] = design
            run.update(evaluation_plan=ref, evaluation_inputs=dependencies)

    def validate_closure(self, run_id, run):
        closure = run["closure"]
        self.contract(closure["checkpoint"], "node-result")
        self.verify_ref(closure["checkpoint"])
        self.markdown(closure["checkpoint"], run_id=run_id, node=run["stage"], search_node_id=run["search_node_id"])
        require(self.value(closure["checkpoint"], "status") in
                ("blocked", "interrupted", "needs_input", "failed", "completed"), "writer still running")
        self.verify_ref(closure["evidence"])
        record = read(self.local(closure["evidence"]["path"]))
        require(record.get("schema") == "operator-harness/run-closure@1" and
                record.get("project_id") == self.project_id and record.get("run_id") == run_id,
                "closure identity mismatch")
        fields(record, "summary")
        require(record.get("writers_stopped") is True and record.get("jobs") == [] and
                record.get("unsettled_resources") == [], "unconfirmed writers, jobs or resources")
        refs = record.get("evidence")
        require(isinstance(refs, list) and bool(refs), "closure needs reconciliation evidence")
        for ref in refs:
            self.verify_ref(ref)
        require(closure["disposition"] in ("rejected", "abandoned", "superseded"), "invalid closure")

    def markdown(self, ref, **expected):
        self.verify_ref(ref)
        if not self.legacy:
            expected = {"project_id": self.project_id, **expected}
        for key, value in expected.items():
            require(self.value(ref, key) == value, key + " must be exactly " + str(value))

    def value(self, ref, key):
        values = metadata(self.local(ref["path"]).read_text(encoding="utf-8"))
        require(key in values and bool(values[key]), "missing/duplicate " + key)
        return values[key]

    def develop_scope(self, task):
        mode = self.value(task, "operation")
        require(mode in ("implement-and-validate", "initial-and-validate", "verify-only", "collect-only", "code-only"),
                "invalid develop operation")
        performance = self.value(task, "performance_required")
        require(performance in ("true", "false"), "performance_required must be true or false")
        require(mode not in ("code-only", "collect-only") or performance == "false",
                "code/diagnostic-only cannot require performance acceptance")
        if "diagnostics_required" in metadata(self.local(task["path"]).read_text()):
            diagnostics = self.value(task, "diagnostics_required")
            require(diagnostics in ("true", "false"), "diagnostics_required must be true or false")
            require(mode != "code-only" or diagnostics == "false", "code-only cannot require device diagnostics")
        return mode, performance == "true"

    def develop_acceptance(self, run, result):
        """Check declared acceptance and freeze evidence; the coordinator reviews its substance."""
        scope = run.get("acceptance_scope", run["task"])
        self.verify_ref(scope)
        mode, performance_required = self.develop_scope(scope)
        self.markdown(result, status="completed")
        accuracy = self.value(result, "accuracy_status")
        performance = self.value(result, "performance_status")
        cause = self.value(result, "regression_cause")
        require(accuracy in ("passed", "not_run", "not_applicable"),
                "accuracy failure/invalidity must be resolved inside develop")
        require(mode in ("code-only", "collect-only") or accuracy == "passed",
                "develop acceptance requires all mandatory accuracy cases to pass")
        require(performance in ("passed", "failed", "not_run", "not_applicable"),
                "invalid performance measurement cannot be accepted")
        require(not performance_required or performance in ("passed", "failed"),
                "required performance has not been measured")
        require(cause in ("none", "idea"),
                "implementation/unknown regression must be resolved inside develop")
        require(performance != "failed" or cause == "idea" or mode == "initial-and-validate",
                "performance failure needs idea diagnosis")
        evidence = {}
        structured = run.get("acceptance_policy", "").endswith("@2")
        if accuracy == "passed":
            evidence["accuracy"] = self.reference(self.value(result, "validation_evidence" if structured else "accuracy_evidence"))
        if performance in ("passed", "failed"):
            require(accuracy == "passed", "performance acceptance requires accuracy passed")
            evidence["performance"] = self.reference(self.value(result, "validation_evidence" if structured else "performance_evidence"))
        if cause == "idea":
            require(accuracy == "passed" and performance in ("passed", "failed"),
                    "idea diagnosis needs valid accuracy and performance results")
            evidence["regression"] = self.reference(self.value(result, "regression_evidence"))
        # Optional for historical TASKs; current templates explicitly declare the scope.
        if "diagnostics_required" in metadata(self.local(scope["path"]).read_text()):
            evidence["analysis_data"] = self.analysis_data_ref(scope, result)
        for ref in evidence.values():
            require(bool(self.local(ref["path"]).read_bytes().strip()), "empty develop evidence")
        return {"policy": "develop-acceptance@1", "evidence": evidence}

    def analysis_data_ref(self, scope, result):
        ref = self.reference(self.value(result, "analysis_data"))
        self.markdown(ref, run_id=self.value(result, "run_id"))
        status = self.value(ref, "status")
        require(status in ("ready", "partial", "unavailable", "not_run", "not_applicable"),
                "invalid analysis data status")
        require(self.value(scope, "diagnostics_required") != "true" or status == "ready",
                "required analysis data is not ready")
        return ref

    def port_scope(self, task):
        self.markdown(task, operation="port-and-validate", aicpu_policy="forbidden")
        required = self.value(task, "performance_required")
        require(required in ("true", "false"), "invalid performance requirement")
        if self.contract(task, "node-task") != "operator-harness/node-task@4":
            require(required == "true", "legacy port requires performance validation")
        require(self.value(task, "diagnostics_required") in ("true", "false"),
                "diagnostics_required must be true or false")

    def port_acceptance(self, run, result):
        """Initial ports may miss the performance target, but must be correct and measured."""
        scope = run["task"]
        self.verify_ref(scope)
        self.port_scope(scope)
        self.markdown(result, status="completed", accuracy_status="passed")
        required = self.value(scope, "performance_required") == "true"
        require(self.value(result, "performance_status") in (("passed", "failed") if required else
                ("passed", "failed", "not_applicable", "not_run")), "port-baseline requires requested measurement")
        structured = run.get("acceptance_policy", "").endswith("@2")
        evidence = {name: self.reference(self.value(result, field)) for name, field in (
            ("accuracy", "validation_evidence" if structured else "accuracy_evidence"),
            ("porting", "porting_evidence"), ("candidate", "candidate_manifest"))}
        if required:
            evidence["performance"] = self.reference(self.value(result, "validation_evidence" if structured else "performance_evidence"))
        candidate = self.value(evidence["candidate"], "candidate_id")
        self.markdown(evidence["candidate"], search_node_id=run["search_node_id"])
        self.markdown(evidence["porting"], contract="operator-harness/porting@1", status="complete",
                      run_id=self.value(result, "run_id"), candidate_id=candidate)
        require(self.value(evidence["porting"], "aicpu_status") in ("not_used", "replaced"),
                "port-baseline must resolve AICPU dependencies")
        evidence["analysis_data"] = self.analysis_data_ref(scope, result)
        self.markdown(evidence["analysis_data"], candidate_id=candidate)
        for ref in evidence.values():
            require(bool(self.local(ref["path"]).read_bytes().strip()), "empty port evidence")
        return {"policy": "port-acceptance@1", "evidence": evidence}

    def run_acceptance(self, run, result):
        expected = "port" if run["stage"] == "port-baseline" else "develop"
        policy = run.get("acceptance_policy", expected + "-acceptance@1")
        require(policy in (expected + "-acceptance@1", expected + "-acceptance@2"), "unsupported acceptance policy")
        accepted = (self.port_acceptance(run, result) if expected == "port"
                    else self.develop_acceptance(run, result))
        if run.get("initial_design"):
            self.verify_ref(run["initial_design"])
            accepted["initial_design"] = run["initial_design"]
        if policy.endswith("@2"):
            self.contract(result, "node-result", required=True)
            mode = self.value(run["task"], "operation")
            accepted["validation"] = (evidence.limited_validation(self, run, result)
                                      if mode in ("code-only", "collect-only")
                                      else evidence.validation(self, run, result))
            accepted["policy"] = policy
        return accepted

    def terminal(self, ref):
        require(self.value(ref, "status") in ("completed", "failed"),
                "only terminal results may be accepted; checkpoints remain outstanding")

    def warnings(self, tree):
        warnings = []
        for binding in tree["cooled_inputs"]:
            ref = binding["artifact"]
            notices = []
            try:
                self.latest_cooled(ref, notices)
                require(read(self.local(ref["path"]))["status"] == "active", "cooled version is not active")
            except Invalid as exc:
                warnings.append({"search_node_id": binding["search_node_id"],
                                 "artifact": ref["path"], "reason": str(exc)})
            warnings.extend({"search_node_id": binding["search_node_id"], **notice} for notice in notices)
        return warnings

    def reconcile(self):
        tree = self.load()
        stale = []
        for key, run in tree["runs"].items():
            if not settled(run):
                for ref in run.get("evaluation_inputs", {}).values():
                    try:
                        self.verify_ref(ref)
                    except (Invalid, OSError) as exc:
                        stale.append({"run_id": key, "path": ref["path"], "error": str(exc)})
        for key, batch in tree.get("batches", {}).items():
            for ref in batch["inputs"].values():
                try:
                    self.verify_ref(ref)
                except (ValueError, OSError) as exc:
                    stale.append({"batch_id": key, "path": ref["path"], "error": str(exc)})
        return {"project_id": self.project_id, "event_watermark": len(tree["events"]), "stale_inputs": stale,
                "corrupted_artifacts": self.audit(tree),
                "best": tree.get("best"), "control": control.snapshot(tree),
                "outstanding_runs": [key for key, run in tree["runs"].items() if not settled(run)],
                "closed_runs": [key for key, run in tree["runs"].items() if run.get("closure")],
                "legacy_acceptances": [key for key, run in tree["runs"].items()
                                       if run.get("result") and not run.get("acceptance", {}).get("validation")],
                "limitations": ["job observations and authority records are supplied by the host; Markdown STATE is not parsed"]}

    def review(self):
        tree = self.load()
        return {"operator": tree["operator"], "task_contract": tree["task_contract"],
                "latest_hot": [{"hot": read(self.local(refs[-1]["path"])), "artifact": refs[-1]}
                               for refs in tree["hot"].values()],
                "instruction": "Select exact revisions at delivery; human approval is required. No automatic promotion."}

    def promote(self, decision, output):
        # The caller records a real human decision; this CLI cannot authenticate people.
        with self.write():
            tree = self.load()
            require(decision["schema"] == "operator-harness/cooling-decision@1", "unsupported decision")
            fields(decision, "id", "reviewer", "reviewed_at", "delivery", "approval_source")
            require(decision["approved"] is True, "explicit human approval required")
            identifier(decision["id"])
            source = self.reference(decision["approval_source"])
            key = identifier(decision["hot_id"])
            revisions = tree["hot"][key]
            revision = decision["revision"]
            require(type(revision) is int and 1 <= revision <= len(revisions), "invalid hot revision")
            ref = revisions[revision - 1]
            self.verify_ref(ref)
            hot = read(self.local(ref["path"]))
            self.validate_hot(hot, tree)
            require(decision["hot_sha256"] == ref["sha256"], "approval does not bind exact hot content")
            fields(decision, "evidence_summary")
            # No local paths or full private approval transcript in the shared package.
            cooled = {"schema": "operator-harness/cooled@1", "id": key,
                      "version": decision["id"], "status": "active",
                      "kind": hot["kind"], "statement": hot["statement"],
                      "applicability": hot["applicability"], "limitations": hot["limitations"],
                      "evidence_summary": decision["evidence_summary"],
                      "provenance": {"operator": tree["operator"], "task_contract": tree["task_contract"],
                                     "hot_id": key, "revision": hot["revision"],
                                     "hot_sha256": ref["sha256"], "nodes": hot["nodes"],
                                     "evidence_sha256": [item["sha256"] for item in hot["evidence"]]},
                      "review": {name: decision[name] for name in ("id", "reviewer", "reviewed_at", "delivery")},
                      "approval_sha256": source["sha256"]}
            validate_cooled(cooled)
            directory = self.local(output)
            require(not directory.is_relative_to((self.root / ".workspace").resolve()),
                    "cooled output must be outside .workspace")
            path = directory / f"{key}--{decision['id']}.json"
            require(path.exists() or revision == len(revisions), "approval refers to stale hot revision")
            with lock(directory):
                items, notices = self.collection(directory)
                require(not notices, "cannot check decision uniqueness: unreadable/non-object collection entries")
                for _, item in items:
                    review = item.get("review")
                    if isinstance(review, dict) and review.get("id") == decision["id"]:
                        require(all(item.get(field) == cooled[field] for field in (
                            "review", "provenance", "approval_sha256", "statement", "kind",
                            "applicability", "limitations", "evidence_summary")),
                            "decision ID already binds a different approval")
                if not path.exists():
                    require(revision == len(revisions), "approval refers to stale hot revision")
                atomic(path, cooled, immutable=True)
            return str(path.relative_to(self.root))

    def collection(self, directory):
        items, notices = [], []
        for path in sorted(directory.glob("*.json")):
            try:
                item = read(path)
                require(isinstance(item, dict), "expected a JSON object")
                items.append((path, item))
            except (OSError, ValueError) as exc:
                notices.append({"artifact": str(path.relative_to(self.root)), "reason": str(exc)})
        return items, notices

    def latest_cooled(self, ref, notices=None):
        # A directory is one knowledge collection; distribute its full history.
        items, problems = self.collection(self.local(ref["path"]).parent)
        if notices is not None:
            notices.extend(problems)
        else:
            for problem in problems:
                print(f"searchctl warning: {problem['artifact']}: {problem['reason']}", file=sys.stderr)
        for _, item in items:
            if item.get("supersedes_sha256") == ref["sha256"]:
                raise Invalid("cooled version has a successor; inspect its status before reuse")

    def invalidate_cooled(self, artifact, status, version, reason):
        with self.write():
            return self._invalidate_cooled(artifact, status, version, reason)

    def _invalidate_cooled(self, artifact, status, version, reason):
        require(status in ("needs_revalidation", "superseded"), "invalidation cannot grant active status")
        identifier(version)
        require(nonempty(reason), "invalidation needs a reason")
        file = self.local(artifact)
        with lock(file.parent):
            ref = self.reference(artifact)
            cooled = read(file)
            validate_cooled(cooled)
            updated = deepcopy(cooled)
            updated.update(version=version, status=status, supersedes_sha256=ref["sha256"], reason=reason)
            path = file.parent / f"{identifier(cooled['id'])}--{version}.json"
            require(path != file, "invalidation must create a new version")
            if not path.exists():
                self.latest_cooled(ref)
            atomic(path, updated, immutable=True)
            return str(path.relative_to(self.root))


def validate_cooled(cooled):
    require(isinstance(cooled, dict), "expected a cooled object")
    require(cooled["schema"] == "operator-harness/cooled@1", "unsupported cooled schema")
    fields(cooled, "id", "version", "statement", "applicability", "limitations", "evidence_summary")
    require(cooled["status"] in ("active", "needs_revalidation", "superseded"), "invalid cooled status")
    require(cooled["kind"] in ("observation", "hypothesis", "conclusion"), "invalid cooled kind")
    fields(cooled["review"], "id", "reviewer", "reviewed_at", "delivery")
    fields(cooled["provenance"], "operator", "task_contract", "hot_id", "hot_sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", "--project-dir", dest="workspace", required=True,
                        help="workspace root, not a source worktree or project record directory")
    parser.add_argument("--project")
    parser.add_argument("--owner")
    parser.add_argument("--legacy", action="store_true", help="read old .workspace/search without migration")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    for flag in ("operator", "task-contract", "hypothesis", "scope"):
        init.add_argument("--" + flag, required=True)
    commands.add_parser("show")
    commands.add_parser("check")
    commands.add_parser("review")
    commands.add_parser("reconcile")
    commands.add_parser("mechanisms")
    commands.add_parser("reference").add_argument("path")
    apply = commands.add_parser("apply")
    apply.add_argument("record", type=Path)
    promote = commands.add_parser("promote")
    promote.add_argument("decision", type=Path)
    promote.add_argument("--output-dir", help="default: knowledge/cooled/<project>")
    invalidate = commands.add_parser("invalidate-cooled")
    invalidate.add_argument("artifact")
    invalidate.add_argument("--status", choices=("needs_revalidation", "superseded"), required=True)
    invalidate.add_argument("--version", required=True)
    invalidate.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        require(not args.legacy or args.command in ("show", "check", "review", "reconcile", "mechanisms", "reference"), "legacy mode is read-only")
        store = Store(args.workspace, args.project, args.owner, args.legacy)
        if args.command == "init":
            result = store.init(args.operator, args.task_contract, args.hypothesis, args.scope)
        elif args.command == "apply":
            result = store.apply(read(args.record))
        elif args.command == "promote":
            result = {"artifact": store.promote(read(args.decision), args.output_dir or f"knowledge/cooled/{store.project_id}")}
        elif args.command == "reference":
            result = store.reference(args.path)
        elif args.command == "mechanisms":
            result = store.mechanisms()
        elif args.command == "reconcile":
            result = store.reconcile()
        elif args.command == "review":
            result = store.review()
        elif args.command == "invalidate-cooled":
            result = {"artifact": store.invalidate_cooled(args.artifact, args.status, args.version, args.reason)}
        else:
            result = store.load()
            warnings = store.warnings(result)
            if args.command == "check":
                issues = store.audit(result)
                result = {"valid": not issues}
                if issues:
                    result["corrupted_artifacts"] = issues
            result["cooled_warnings"] = warnings
        print(encoded(result).decode(), end="")
        if args.command == "check" and not result["valid"]:
            return 1
        return 0
    except (Invalid, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"searchctl: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
