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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness-new-project/scripts"))
import projectctl


class Invalid(ValueError):
    pass


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
        require(isinstance(tree, dict), "expected a search object")
        require(tree["schema"] == "operator-harness/search@1", "unsupported search schema")
        if not self.legacy:
            require(tree.get("project_id") == self.project_id, "search project identity mismatch")
        fields(tree, "operator", "task_contract")
        nodes = tree["nodes"]
        require(tree["current_node"] in nodes, "unknown current node")
        roots = [key for key, node in nodes.items() if node["parent"] is None]
        require(len(roots) == 1, "tree must have one root")
        for key, node in nodes.items():
            identifier(key)
            fields(node, "hypothesis", "scope")
            require(node["status"] in ("open", "parked", "closed"), "invalid node status")
            seen = set()
            cursor = key
            while cursor is not None:
                require(cursor in nodes and cursor not in seen, "missing parent or tree cycle")
                seen.add(cursor)
                cursor = nodes[cursor]["parent"]
        for key, candidate in tree["candidates"].items():
            identifier(key)
            require(candidate["search_node_id"] in nodes, "unknown candidate node")
            self.verify_ref(candidate["manifest"])
            self.markdown(candidate["manifest"], candidate_id=key,
                          search_node_id=candidate["search_node_id"])
            seen = set()
            cursor = key
            while cursor is not None:
                require(cursor in tree["candidates"] and cursor not in seen, "invalid candidate lineage")
                seen.add(cursor)
                cursor = tree["candidates"][cursor]["parent_candidate"]
            parent = candidate["parent_candidate"]
            self.markdown(candidate["manifest"], parent_candidate=(
                tree["candidates"][parent]["manifest"]["path"] if parent else "not_applicable"))
        for key, run in tree["runs"].items():
            identifier(key)
            require(run["search_node_id"] in nodes, "unknown run node")
            require(run["stage"] in ("analysis", "develop", "device-test"), "invalid workflow stage")
            self.verify_ref(run["task"])
            self.markdown(run["task"], run_id=key, node=run["stage"],
                          search_node_id=run["search_node_id"])
            candidate = run["candidate_id"]
            require(candidate is None or candidate in tree["candidates"], "unknown input candidate")
            if run.get("result"):
                self.verify_ref(run["result"])
                self.markdown(run["result"], run_id=key, node=run["stage"],
                              search_node_id=run["search_node_id"])
                self.terminal(run["result"])
                if "acceptance" in run:
                    require(run["acceptance"] == self.develop_acceptance(run, run["result"]),
                            "develop acceptance evidence changed")
            if run.get("batch_id"):
                require(run["batch_id"] in tree.get("batches", {}), "unknown develop batch")
                require(key in tree["batches"][run["batch_id"]]["members"], "unlisted batch run")
        for key, batch in tree.get("batches", {}).items():
            identifier(key)
            require(len(batch["members"]) >= 2 and
                    len(set(batch["members"])) == len(batch["members"]), "invalid batch members")
            require(set(batch["inputs"]) == {"requirements", "baseline", "cases", "protocol", "environment"},
                    "batch needs frozen comparison inputs")
            for ref in batch["inputs"].values():
                self.verify_ref(ref)
            repositories = set()
            for member in batch["members"]:
                run = tree["runs"][member]
                require(run.get("batch_id") == key and run["stage"] == "develop", "batch run mismatch")
                require(run["candidate_id"] == batch["candidate_id"], "batch candidate mismatch")
                require(run["analysis_run"] in tree["runs"], "unknown analysis run")
                analysis = tree["runs"][run["analysis_run"]]
                require(analysis["stage"] == "analysis" and analysis.get("result") and
                        analysis["candidate_id"] == run["candidate_id"] and
                        analysis["search_node_id"] == run["search_node_id"], "inapplicable analysis run")
                self.markdown(analysis["result"], status="completed")
                require(bool(run["repo_ids"]) and len(set(run["repo_ids"])) == len(run["repo_ids"]),
                        "invalid batch repositories")
                require(not repositories.intersection(run["repo_ids"]), "batch worktrees overlap")
                repositories.update(run["repo_ids"])
                self.markdown(run["task"], batch_id=key, analysis_run=run["analysis_run"])
                if run.get("result"):
                    self.markdown(run["result"], batch_id=key, analysis_run=run["analysis_run"])
        for key, revisions in tree["hot"].items():
            identifier(key)
            require(bool(revisions), "empty hot history")
            for revision, ref in enumerate(revisions, 1):
                self.verify_ref(ref)
                hot = read(self.local(ref["path"]))
                require(hot["id"] == key and hot["revision"] == revision, "invalid hot revision")
                self.validate_hot(hot, tree)
        for binding in tree["cooled_inputs"]:
            require(binding["search_node_id"] in nodes, "unknown knowledge consumer")
            fields(binding, "reason")
            self.verify_ref(binding["artifact"])
            validate_cooled(read(self.local(binding["artifact"]["path"])))

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
        with self.write(closing=operation.get("op") == "accept"):
            tree = self.load()
            event_id = identifier(operation["event_id"])
            for event in tree["events"]:
                if event["operation"]["event_id"] == event_id:
                    require(event["operation"] == operation, "event ID reused with different content")
                    return tree
            self.mutate(tree, operation)
            self.validate(tree)
            tree["events"].append({"at": now(), "operation": deepcopy(operation)})
            atomic(self.path, tree)
            return tree

    def mutate(self, tree, op):
        action = op["op"]
        if action == "add-node":
            key = identifier(op["id"])
            require(key not in tree["nodes"], "node already exists")
            require(op["parent"] in tree["nodes"], "unknown parent")
            fields(op, "hypothesis", "scope")
            tree["nodes"][key] = {name: op[name] for name in ("parent", "hypothesis", "scope")}
            tree["nodes"][key]["status"] = "open"
        elif action in ("select", "node-status"):
            require(op["id"] in tree["nodes"], "unknown node")
            fields(op, "reason")
            if action == "select":
                require(tree["nodes"][op["id"]]["status"] == "open", "reopen node before selecting")
                require(all(run.get("result") for run in tree["runs"].values()),
                        "accept outstanding runs before switching strategy")
                tree["current_node"] = op["id"]
            else:
                require(op["status"] in ("open", "parked", "closed"), "invalid status")
                require(op["status"] == "open" or all(
                    run.get("result") or run["search_node_id"] != op["id"]
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
            self.markdown(ref, parent_candidate=(tree["candidates"][parent]["manifest"]["path"]
                                                if parent else "not_applicable"))
            tree["candidates"][key] = {"search_node_id": op["search_node_id"],
                                       "parent_candidate": parent, "manifest": ref}
        elif action == "run":
            require(op["stage"] in ("analysis", "develop"),
                    "new runs use analysis or develop; device-test is legacy only")
            require(not any(run.get("batch_id") and not run.get("result")
                            for run in tree["runs"].values()), "develop batch is outstanding")
            require(not any(not run.get("result") and
                            (op["stage"] == "develop" or run["stage"] != "analysis")
                            for run in tree["runs"].values()),
                    "ordinary develop must run alone; use develop-batch for parallel work")
            key = identifier(op["id"])
            require(key not in tree["runs"], "run already registered")
            require(op["search_node_id"] == tree["current_node"], "run must use selected node")
            require(tree["nodes"][tree["current_node"]]["status"] == "open", "node is not open")
            ref = self.reference(op["task"])
            self.markdown(ref, run_id=key, node=op["stage"], search_node_id=op["search_node_id"])
            if op["stage"] == "develop":
                self.develop_scope(ref)
            tree["runs"][key] = {"search_node_id": op["search_node_id"], "stage": op["stage"],
                                  "candidate_id": op["candidate_id"], "task": ref, "result": None}
            if op["stage"] == "develop":
                tree["runs"][key]["acceptance_policy"] = "develop-acceptance@1"
        elif action == "develop-batch":
            self.develop_batch(tree, op)
        elif action == "accept":
            run = tree["runs"][op["id"]]
            require(run["result"] is None, "run already accepted")
            ref = self.reference(op["result"])
            self.markdown(ref, run_id=op["id"], node=run["stage"], search_node_id=run["search_node_id"])
            self.terminal(ref)
            if run["stage"] == "develop":
                if op.get("scope"):
                    require("acceptance_policy" not in run, "new develop cannot replace its frozen scope")
                    run["acceptance_scope"] = self.reference(op["scope"])
                    self.markdown(run["acceptance_scope"], run_id=op["id"])
                run["acceptance"] = self.develop_acceptance(run, ref)
            run["result"] = ref
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
        require(all(run.get("result") for run in tree["runs"].values()),
                "accept outstanding runs before starting a batch")
        members = op["members"]
        require(isinstance(members, list) and len(members) >= 2, "batch needs at least two members")
        require(isinstance(op["inputs"], dict), "batch inputs must be an object")
        inputs = {name: self.reference(path) for name, path in op["inputs"].items()}
        data = self.workspace.load(self.project_id)
        paths = []
        ids = []
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
            tree["runs"][run_id] = {"search_node_id": node, "stage": "develop",
                                    "candidate_id": op["candidate_id"], "task": task, "result": None,
                                    "batch_id": key, "analysis_run": member["analysis_run"],
                                    "acceptance_policy": "develop-acceptance@1",
                                    "repo_ids": list(repo_ids)}
            ids.append(run_id)
        tree.setdefault("batches", {})[key] = {"members": ids, "inputs": inputs,
                                               "candidate_id": op["candidate_id"]}

    def markdown(self, ref, **expected):
        text = self.local(ref["path"]).read_text(encoding="utf-8")
        if not self.legacy:
            expected = {"project_id": self.project_id, **expected}
        for key, value in expected.items():
            values = re.findall(rf"^- {re.escape(key)}: (.*?)\s*$", text, re.M)
            if key == "parent_candidate" and value == "not_applicable":
                values = [re.sub(r"^not_applicable\s*[:：]\s*.+$", "not_applicable", item)
                          for item in values]
            require(values == [value], f"manifest {key} must be exactly {value}")

    def value(self, ref, key):
        text = self.local(ref["path"]).read_text(encoding="utf-8")
        values = re.findall(rf"^- {re.escape(key)}: ([^\n\r]*)$", text, re.M)
        require(len(values) == 1 and nonempty(values[0]), f"missing/duplicate {key}")
        return values[0].strip()

    def develop_scope(self, task):
        mode = self.value(task, "operation")
        require(mode in ("implement-and-validate", "verify-only", "collect-only", "code-only"),
                "invalid develop operation")
        performance = self.value(task, "performance_required")
        require(performance in ("true", "false"), "performance_required must be true or false")
        require(mode not in ("code-only", "collect-only") or performance == "false",
                "code/diagnostic-only cannot require performance acceptance")
        if re.search(r"^- diagnostics_required:", self.local(task["path"]).read_text(), re.M):
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
        require(performance != "failed" or cause == "idea", "performance failure needs idea diagnosis")
        evidence = {}
        if accuracy == "passed":
            evidence["accuracy"] = self.reference(self.value(result, "accuracy_evidence"))
        if performance in ("passed", "failed"):
            require(accuracy == "passed", "performance acceptance requires accuracy passed")
            evidence["performance"] = self.reference(self.value(result, "performance_evidence"))
        if cause == "idea":
            require(accuracy == "passed" and performance in ("passed", "failed"),
                    "idea diagnosis needs valid accuracy and performance results")
            evidence["regression"] = self.reference(self.value(result, "regression_evidence"))
        # Optional for historical TASKs; current templates explicitly declare the scope.
        if re.search(r"^- diagnostics_required:", self.local(scope["path"]).read_text(), re.M):
            ref = self.reference(self.value(result, "analysis_data"))
            self.markdown(ref, run_id=self.value(result, "run_id"))
            status = self.value(ref, "status")
            require(status in ("ready", "partial", "unavailable", "not_run", "not_applicable"),
                    "invalid analysis data status")
            require(self.value(scope, "diagnostics_required") != "true" or status == "ready",
                    "required analysis data is not ready")
            evidence["analysis_data"] = ref
        for ref in evidence.values():
            require(bool(self.local(ref["path"]).read_bytes().strip()), "empty develop evidence")
        return {"policy": "develop-acceptance@1", "evidence": evidence}

    def terminal(self, ref):
        text = self.local(ref["path"]).read_text(encoding="utf-8")
        values = re.findall(r"^- status: (.*?)[ \t]*$", text, re.M)
        require(values in (["completed"], ["failed"]),
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
            hot = read(self.local(ref["path"]))
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
        require(not args.legacy or args.command in ("show", "check", "review"), "legacy mode is read-only")
        store = Store(args.workspace, args.project, args.owner, args.legacy)
        if args.command == "init":
            result = store.init(args.operator, args.task_contract, args.hypothesis, args.scope)
        elif args.command == "apply":
            result = store.apply(read(args.record))
        elif args.command == "promote":
            result = {"artifact": store.promote(read(args.decision), args.output_dir or f"knowledge/cooled/{store.project_id}")}
        elif args.command == "review":
            result = store.review()
        elif args.command == "invalidate-cooled":
            result = {"artifact": store.invalidate_cooled(args.artifact, args.status, args.version, args.reason)}
        else:
            result = store.load()
            warnings = store.warnings(result)
            if args.command == "check":
                result = {"valid": True}
            result["cooled_warnings"] = warnings
        print(encoded(result).decode(), end="")
    except (Invalid, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"searchctl: {exc}\n")


if __name__ == "__main__":
    main()
