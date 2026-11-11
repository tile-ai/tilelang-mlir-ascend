"""CPU-only integration checks for search records and knowledge promotion."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "assets/skills/harness-resume/scripts/searchctl.py"
sys.path.insert(0, str(SCRIPT.parent))
import searchctl as search


class SearchToolsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        search.projectctl.Workspace(self.root).create("example", "Example", "session")
        self.store = search.Store(self.root, "example", "session")
        self.store.init("example", "requirements-v1", "initial implementation", "all cases")
        self.sequence = 0

    def write(self, path, text):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        return path

    def apply(self, action, **values):
        self.sequence += 1
        if action in ("run", "develop-batch"):
            values.setdefault("legacy", True)
        return self.store.apply({"event_id": f"e{self.sequence}", "op": action, **values})

    def add_node(self, key="N1", parent="N0"):
        return self.apply("add-node", id=key, parent=parent, hypothesis="reuse loads", scope="large cases")

    def candidate(self, key, node="N0", parent=None):
        path = f".workspace/example/candidates/{key}.md"
        parent_path = f".workspace/example/candidates/{parent}.md" if parent else "not_applicable"
        self.write(path, f"- project_id: example\n- candidate_id: {key}\n- search_node_id: {node}\n- parent_candidate: {parent_path}\n")
        return self.apply("candidate", id=key, search_node_id=node, parent_candidate=parent, manifest=path)

    def record_run(self, key, stage="analysis", node="N0", candidate=None, finish=True):
        text = f"- project_id: example\n- run_id: {key}\n- node: {stage}\n- search_node_id: {node}\n"
        if stage == "develop":
            text += "- operation: implement-and-validate\n- performance_required: true\n"
        task = self.write(f".workspace/example/runs/{key}/TASK.md", text)
        self.apply("run", id=key, search_node_id=node, stage=stage, candidate_id=candidate, task=task)
        if finish:
            result = self.write(f".workspace/example/runs/{key}/RESULT.md", text + "- status: completed\n" +
                                (self.acceptance_fields(key) if stage == "develop" else ""))
            self.apply("accept", id=key, result=result)

    def hot(self, revision=0, nodes=None):
        evidence = self.write(f".workspace/example/evidence/{revision}.md", f"measured sample {revision}\n")
        return self.apply("hot", id="H1", base_revision=revision, kind="observation",
                          statement="load duration decreased", applicability="synthetic test conditions",
                          limitations="end-to-end benefit unverified", nodes=nodes or ["N0"], evidence=[evidence])

    def decision(self):
        ref = self.store.load()["hot"]["H1"][-1]
        hot = search.read(self.store.local(ref["path"]))
        approval = self.write(".workspace/example/decisions/review.md", "Synthetic approval fixture, not a real approval.\n")
        return {"schema": "operator-harness/cooling-decision@1", "id": "review1", "approved": True,
                "reviewer": "test-reviewer", "reviewed_at": "2026-01-01T00:00:00Z", "delivery": "test-delivery",
                "approval_source": approval, "hot_id": "H1", "revision": hot["revision"],
                "hot_sha256": ref["sha256"], "evidence_summary": "Synthetic fixture with no device claims."}

    def test_symlinked_workspace_cannot_receive_cooled(self):
        self.hot()
        decision = self.decision()
        (self.root / ".workspace").rename(self.root / "private")
        (self.root / ".workspace").symlink_to("private", target_is_directory=True)
        for output in (".workspace/cooled", "private/cooled"):
            with self.subTest(output=output), self.assertRaisesRegex(search.Invalid, "outside .workspace"):
                self.store.promote(decision, output)
        self.assertFalse((self.root / "private/cooled").exists())

    def test_decision_id_cannot_be_reused_for_another_hot_entry(self):
        self.hot()
        decision = self.decision()
        self.store.promote(decision, "knowledge/cooled")
        self.apply("hot", id="H2", base_revision=0, kind="observation", statement="other finding",
                   applicability="all", limitations="test only", nodes=["N0"],
                   evidence=[".workspace/example/evidence/0.md"])
        ref = self.store.load()["hot"]["H2"][-1]
        with self.assertRaisesRegex(search.Invalid, "decision ID"):
            self.store.promote({**decision, "hot_id": "H2", "hot_sha256": ref["sha256"]}, "knowledge/cooled")
        self.assertFalse((self.root / "knowledge/cooled/H2--review1.json").exists())

    def test_invalid_collection_neighbors_warn_without_breaking_commands(self):
        self.hot()
        artifact = self.store.promote(self.decision(), "knowledge/cooled")
        self.write("knowledge/cooled/array.json", "[]")
        self.write("knowledge/cooled/broken.json", "{")
        self.write("knowledge/cooled/unrelated.json", '{"other": true}')
        with patch("sys.stderr") as stderr:
            self.apply("use-cooled", search_node_id="N0", artifact=artifact, reason="matching scope")
            self.assertTrue(stderr.write.called)
        for command in ("show", "check"):
            result = self.cli(command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)["cooled_warnings"]), 2)
        with patch("sys.stderr"):
            self.store.invalidate_cooled(artifact, "needs_revalidation", "v2", "new toolchain")
        result = self.cli("show")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["cooled_warnings"]), 3)
        with patch("sys.stderr"), self.assertRaisesRegex(search.Invalid, "successor"):
            self.apply("use-cooled", search_node_id="N0", artifact=artifact, reason="reuse")

    def test_hot_evidence_snapshots_survive_live_result_changes(self):
        live = self.write(".workspace/example/runs/live/RESULT.md", "checkpoint one")
        values = dict(id="H1", kind="observation", statement="partial observation",
                      applicability="test", limitations="incomplete", nodes=["N0"], evidence=[live])
        self.apply("hot", base_revision=0, **values)
        first = self.store.review()["latest_hot"][0]["hot"]["evidence"][0]
        self.write(live, "checkpoint two")
        self.apply("hot", base_revision=1, **values)
        self.assertEqual(self.store.local(first["path"]).read_text(), "checkpoint one")
        second = self.store.review()["latest_hot"][0]["hot"]["evidence"][0]
        self.assertEqual(self.store.local(second["path"]).read_text(), "checkpoint two")
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.store.promote(self.decision(), "knowledge/cooled")
        self.write(first["path"], "tampered snapshot")
        tree = self.store.load()
        self.assertIn(first["path"], [item["path"] for item in self.store.audit(tree)])
        self.assertNotEqual(self.cli("check").returncode, 0)
        self.assertEqual(self.cli("show").returncode, 0)
        # Historical damage does not prevent promotion of the intact latest revision.
        self.store.promote(self.decision(), "knowledge/cooled")
        self.write(second["path"], "tampered current snapshot")
        with self.assertRaisesRegex(search.Invalid, "artifact changed"):
            self.store.promote(self.decision(), "knowledge/cooled")

    def test_cooling_rechecks_approved_hot_content(self):
        self.hot()
        decision = self.decision()
        ref = self.store.load()["hot"]["H1"][-1]
        hot = search.read(self.store.local(ref["path"]))
        hot["statement"] = "changed after approval"
        self.write(ref["path"], json.dumps(hot))
        self.assertTrue(self.store.audit(self.store.load()))
        with self.assertRaisesRegex(search.Invalid, "artifact changed"):
            self.store.promote(decision, "knowledge/cooled")
        self.assertFalse((self.root / "knowledge/cooled/H1--review1.json").exists())

    def test_hot_snapshots_are_isolated_by_project(self):
        self.hot()
        first = self.store.review()["latest_hot"][0]["hot"]["evidence"][0]
        search.projectctl.Workspace(self.root).create("other", "Other", "session")
        other = search.Store(self.root, "other", "session")
        other.init("other", "requirements-v1", "initial", "all")
        source = self.write(".workspace/other/evidence.md", "measured sample 0\n")
        other.apply({"event_id": "e1", "op": "hot", "id": "H1", "base_revision": 0,
                     "kind": "observation", "statement": "test", "applicability": "test",
                     "limitations": "test", "nodes": ["N0"], "evidence": [source]})
        second = other.review()["latest_hot"][0]["hot"]["evidence"][0]
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(first["path"], second["path"])
        self.assertTrue(first["path"].startswith(".workspace/example/knowledge/evidence/"))
        self.assertTrue(second["path"].startswith(".workspace/other/knowledge/evidence/"))
        with self.assertRaisesRegex(search.Invalid, "another project"):
            other.snapshot(first["path"])

    def test_root_candidate_accepts_reason_suffix(self):
        for index, parent in enumerate(("not_applicable", "not_applicable：初版", "not_applicable: initial")):
            key = f"C{index}"
            path = self.write(f"{key}.md", f"- project_id: example\n- candidate_id: {key}\n- search_node_id: N0\n- parent_candidate: {parent}\n")
            self.apply("candidate", id=key, search_node_id="N0", parent_candidate=None, manifest=path)
        self.assertEqual(len(self.store.load()["candidates"]), 3)


    def acceptance_fields(self, key, accuracy="passed", performance="passed", cause="none"):
        paths = {kind: self.write(f".workspace/example/runs/{key}/{kind}.md",
                                 f"Synthetic {kind} evidence for {key}; no device claims.\n")
                 for kind in ("accuracy", "performance", "regression")}
        return (f"- accuracy_status: {accuracy}\n- performance_status: {performance}\n"
                f"- regression_cause: {cause}\n" + "".join(
                    f"- {kind}_evidence: {path}\n" for kind, path in paths.items()))

    def batch_fixture(self):
        source = self.root / "source"
        source.mkdir()
        for args in (("init",), ("config", "user.name", "Fixture"),
                     ("config", "user.email", "fixture@example.invalid"),
                     ("commit", "--allow-empty", "-m", "fixture")):
            subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
        self.candidate("C0")
        self.record_run("a0", candidate="C0")
        self.add_node()
        self.apply("select", id="N1", reason="second proposal")
        self.record_run("a1", node="N1", candidate="C0")
        members = []
        for index in range(2):
            key, node, analysis, repo = f"d{index}", f"N{index}", f"a{index}", f"variant{index}"
            self.store.workspace.add_repo("example", "session", repo, source, "HEAD")
            task = self.write(f".workspace/example/runs/{key}/TASK.md",
                              f"- project_id: example\n- run_id: {key}\n- node: develop\n"
                              f"- search_node_id: {node}\n- batch_id: B1\n- analysis_run: {analysis}\n"
                              "- operation: implement-and-validate\n- performance_required: true\n")
            members.append({"id": key, "search_node_id": node, "analysis_run": analysis,
                            "repo_ids": [repo], "task": task})
        inputs = {name: self.write(f".workspace/example/inputs/{name}.md", name)
                  for name in ("requirements", "baseline", "cases", "protocol", "environment")}
        return {"event_id": "batch-event", "op": "develop-batch", "legacy": True, "id": "B1",
                "candidate_id": "C0", "inputs": inputs, "members": members}

    def finish_member(self, key, status="completed"):
        task = self.root / f".workspace/example/runs/{key}/TASK.md"
        result = self.write(f".workspace/example/runs/{key}/RESULT.md",
                            task.read_text() + f"- status: {status}\n" + self.acceptance_fields(key))
        return self.apply("accept", id=key, result=result)

    def cli(self, *args, script=SCRIPT):
        return subprocess.run([sys.executable, str(script), "--workspace", str(self.root),
                               "--project", "example", "--owner", "session", *args],
                              text=True, capture_output=True)

    def test_tree_lineage_outstanding_runs_and_idempotency(self):
        self.candidate("C0")
        self.add_node()
        self.add_node("N2", "N1")
        self.apply("select", id="N2", reason="independent layout strategy")
        self.candidate("C2", node="N2", parent="C0")
        self.record_run("branch-test", "develop", "N2", "C2")
        self.apply("select", id="N0", reason="revisit initial hypothesis")
        self.record_run("r1", finish=False)
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.apply("select", id="N1", reason="switch")
        result = self.write(".workspace/example/runs/r1/RESULT.md",
                            "- project_id: example\n- run_id: r1\n- node: analysis\n- search_node_id: N0\n- status: interrupted\n")
        with self.assertRaisesRegex(search.Invalid, "terminal"):
            self.apply("accept", id="r1", result=result)
        operation = {"event_id": "stable", "op": "add-node", "id": "N3", "parent": "N0",
                     "hypothesis": "reuse", "scope": "all"}
        self.store.apply(operation)
        before = self.store.path.read_bytes()
        self.store.apply(operation)
        self.assertEqual(self.store.path.read_bytes(), before)
        with self.assertRaisesRegex(search.Invalid, "reused"):
            self.store.apply({**operation, "scope": "other"})
        tree = self.store.load()
        tree["nodes"]["N1"]["parent"] = "N2"
        with self.assertRaisesRegex(search.Invalid, "cycle"):
            self.store.validate(tree)

    def test_hot_knowledge_cooling_and_invalidation_lifecycle(self):
        self.add_node()
        self.hot(nodes=["N0", "N1"])
        decision = self.decision()
        for changes in ({"approved": False}, {"hot_sha256": "0" * 64}, {"revision": 99}, {"reviewer": ""}):
            with self.subTest(changes=changes), self.assertRaises(search.Invalid):
                self.store.promote({**decision, **changes}, "knowledge/cooled")
        path = self.store.promote(decision, "knowledge/cooled")
        self.assertEqual(self.store.promote(decision, "knowledge/cooled"), path)
        with self.assertRaisesRegex(search.Invalid, "decision ID"):
            self.store.promote({**decision, "evidence_summary": "different claim"}, "knowledge/cooled")
        self.assertNotIn(".workspace", (self.root / path).read_text())
        self.apply("use-cooled", search_node_id="N0", artifact=path, reason="matching conditions")
        new = self.store.invalidate_cooled(path, "needs_revalidation", "v2", "toolchain changed")
        for artifact in (path, new):
            with self.subTest(artifact=artifact), self.assertRaises(search.Invalid):
                self.apply("use-cooled", search_node_id="N0", artifact=artifact, reason="reuse")
        self.assertEqual(len(json.loads(self.cli("show").stdout)["cooled_warnings"]), 1)

    def test_paths_read_only_commands_and_concurrent_writers(self):
        for path in ("../outside", str(self.root / "absolute")):
            with self.subTest(path=path), self.assertRaises(search.Invalid):
                self.store.local(path)
        before = self.store.path.read_bytes()
        for command in ("show", "check", "review"):
            self.assertEqual(self.cli(command).returncode, 0)
        self.assertEqual(self.store.path.read_bytes(), before)
        processes = []
        for key in ("N1", "N2"):
            file = self.root / f"{key}.json"
            file.write_text(json.dumps({"event_id": key, "op": "add-node", "id": key,
                                        "parent": "N0", "hypothesis": key, "scope": "all"}))
            processes.append(subprocess.Popen(
                [sys.executable, str(SCRIPT), "--workspace", str(self.root),
                 "--project", "example", "--owner", "session", "apply", str(file)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            _, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, error)
        self.assertEqual(set(self.store.load()["nodes"]), {"N0", "N1", "N2"})

    def test_batch_isolation_and_legacy_compatibility(self):
        operation = self.batch_fixture()
        for field in ("repo_ids", "task", "id"):
            bad = deepcopy(operation)
            bad["members"][1][field] = bad["members"][0][field]
            with self.assertRaises(search.Invalid):
                self.store.apply(bad)
        self.store.apply(operation)
        self.finish_member("d1")
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.apply("select", id="N0", reason="too early")
        self.finish_member("d0")
        self.apply("select", id="N0", reason="all results retained")
        port_op, port_res = self._legacy_port_fixture()
        self.apply("run", **port_op)
        self.apply("accept", id="port", result=port_res)
        self.assertEqual(self.store.load()["runs"]["port"]["acceptance"]["policy"], "port-acceptance@1")

    def _legacy_port_fixture(self):
        prefix = ".workspace/example/runs/port"
        identity = "- project_id: example\n- run_id: port\n- node: port-baseline\n- search_node_id: N0\n"
        task = self.write(prefix + "/TASK.md", identity +
                          "- operation: port-and-validate\n- performance_required: true\n"
                          "- diagnostics_required: true\n- aicpu_policy: forbidden\n")
        candidate = self.write(prefix + "/CANDIDATE.md",
                               "- project_id: example\n- candidate_id: Cport\n- search_node_id: N0\n"
                               "- parent_candidate: not_applicable\n")
        mapping = self.write(prefix + "/PORTING.md", "- contract: operator-harness/porting@1\n"
                             "- project_id: example\n- run_id: port\n- candidate_id: Cport\n"
                             "- status: complete\n- aicpu_status: replaced\nSynthetic replacement evidence.\n")
        data = self.write(prefix + "/ANALYSIS_DATA.md", "- project_id: example\n- run_id: port\n"
                          "- candidate_id: Cport\n- status: ready\nSynthetic CPU fixture only.\n")
        accuracy = self.write(prefix + "/accuracy.md", "Synthetic mandatory case results.\n")
        performance = self.write(prefix + "/performance.md", "Synthetic valid but slow sample.\n")
        result = self.write(prefix + "/RESULT.md", identity + "- status: completed\n- accuracy_status: passed\n"
                            f"- accuracy_evidence: {accuracy}\n- performance_status: failed\n"
                            f"- performance_evidence: {performance}\n- candidate_manifest: {candidate}\n"
                            f"- porting_evidence: {mapping}\n- analysis_data: {data}\n")
        return dict(id="port", search_node_id="N0", stage="port-baseline", candidate_id=None, task=task), result


if __name__ == "__main__":
    unittest.main()
