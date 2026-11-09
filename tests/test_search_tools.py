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

    def test_many_loops_and_candidates_in_one_strategy_and_return_to_branch(self):
        self.candidate("C0")
        for iteration in range(2):
            for stage in ("analysis", "develop"):
                self.record_run(f"r{iteration}-{stage}", stage, candidate="C0")
        self.candidate("C1", parent="C0")
        self.add_node()
        self.add_node("N2", "N1")
        self.apply("select", id="N2", reason="independent layout strategy")
        self.candidate("C2", node="N2", parent="C0")
        self.record_run("branch-test", "develop", "N2", "C2")
        self.apply("select", id="N0", reason="revisit initial hypothesis")
        self.record_run("retest", "develop", candidate="C1")
        tree = search.Store(self.root, "example").load()
        self.assertEqual(len(tree["nodes"]), 3)
        self.assertEqual(len(tree["runs"]), 6)
        self.assertEqual(tree["candidates"]["C2"]["parent_candidate"], "C0")
        self.assertEqual(tree["current_node"], "N0")

    def test_outstanding_run_blocks_switch_and_checkpoint_is_not_terminal(self):
        self.add_node()
        self.record_run("r1", finish=False)
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.apply("select", id="N1", reason="switch")
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.apply("node-status", id="N0", status="closed", reason="done")
        result = self.write(".workspace/example/runs/r1/RESULT.md",
                            "- project_id: example\n- run_id: r1\n- node: analysis\n- search_node_id: N0\n- status: interrupted\n")
        with self.assertRaisesRegex(search.Invalid, "terminal"):
            self.apply("accept", id="r1", result=result)
        self.assertIsNone(self.store.load()["runs"]["r1"]["result"])

    def test_retries_are_idempotent_but_conflicting_event_ids_fail(self):
        operation = {"event_id": "stable", "op": "add-node", "id": "N1", "parent": "N0",
                     "hypothesis": "reuse", "scope": "all"}
        self.store.apply(operation)
        before = self.store.path.read_bytes()
        self.store.apply(operation)
        self.assertEqual(self.store.path.read_bytes(), before)
        with self.assertRaisesRegex(search.Invalid, "reused"):
            self.store.apply({**operation, "scope": "other"})
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_bad_parent_and_cycles_do_not_commit(self):
        with self.assertRaisesRegex(search.Invalid, "parent"):
            self.add_node(parent="missing")
        self.add_node()
        self.add_node("N2", "N1")
        tree = self.store.load()
        tree["nodes"]["N1"]["parent"] = "N2"
        with self.assertRaisesRegex(search.Invalid, "cycle"):
            self.store.validate(tree)
        self.assertEqual(self.store.load()["nodes"]["N1"]["parent"], "N0")

    def test_manifest_mismatch_and_tampering_fail(self):
        self.candidate("C0")
        path = self.write("wrong.md", "- project_id: example\n- candidate_id: C1\n- search_node_id: N2\n")
        with self.assertRaisesRegex(search.Invalid, "search_node_id"):
            self.apply("candidate", id="C1", search_node_id="N0", parent_candidate=None, manifest=path)
        self.write(".workspace/example/candidates/C0.md", "changed after registration\n")
        with self.assertRaisesRegex(search.Invalid, "artifact changed"):
            self.store.load()

    def test_hot_revisions_multi_node_evidence_and_stale_updates(self):
        self.add_node()
        self.hot(nodes=["N0", "N1"])
        first = deepcopy(self.store.load()["hot"]["H1"][0])
        self.hot(1, ["N1"])
        with self.assertRaisesRegex(search.Invalid, "stale"):
            self.hot(0)
        self.assertEqual(len(self.store.load()["hot"]["H1"]), 2)
        self.assertEqual(search.read(self.store.local(first["path"]))["nodes"], ["N0", "N1"])
        self.assertEqual(self.store.review()["latest_hot"][0]["hot"]["revision"], 2)

    def test_hot_crash_before_index_commit_can_resume_without_duplicate(self):
        original = search.atomic

        def interrupt_index(path, value, immutable=False):
            if path == self.store.path:
                raise OSError("simulated interruption")
            return original(path, value, immutable)

        with patch.object(search, "atomic", side_effect=interrupt_index):
            with self.assertRaisesRegex(OSError, "interruption"):
                self.hot()
        self.assertEqual(self.store.load()["hot"], {})
        self.hot()
        self.assertEqual(len(list((self.root / ".workspace/example/knowledge/hot/H1").glob("*.json"))), 1)
        self.assertEqual(len(self.store.load()["hot"]["H1"]), 1)

    def test_promotion_requires_exact_approved_revision_and_retains_hot(self):
        self.hot()
        decision = self.decision()
        for changes in ({"approved": False}, {"hot_sha256": "0" * 64}, {"revision": 99}, {"reviewer": ""}):
            with self.subTest(changes=changes), self.assertRaises(search.Invalid):
                self.store.promote({**decision, **changes}, "knowledge/cooled")
        output = self.store.promote(decision, "knowledge/cooled")
        before = (self.root / output).read_bytes()
        self.assertEqual(self.store.promote(decision, "knowledge/cooled"), output)
        self.assertEqual((self.root / output).read_bytes(), before)
        cooled = search.read(self.root / output)
        self.assertNotIn(".workspace", before.decode())
        self.assertEqual(cooled["provenance"]["hot_sha256"], decision["hot_sha256"])
        self.assertIn("H1", self.store.load()["hot"])
        self.hot(1)
        self.assertEqual(self.store.promote(decision, "knowledge/cooled"), output)
        with self.assertRaisesRegex(search.Invalid, "decision ID"):
            self.store.promote({**decision, "evidence_summary": "different claim"}, "knowledge/cooled")

    def test_hot_update_invalidates_old_approval(self):
        self.hot()
        decision = self.decision()
        self.hot(1)
        with self.assertRaisesRegex(search.Invalid, "stale"):
            self.store.promote(decision, "knowledge/cooled")
        self.assertFalse((self.root / "knowledge").exists())

    def test_invalidation_keeps_history_and_prevents_new_consumption(self):
        self.hot()
        path = self.store.promote(self.decision(), "knowledge/cooled")
        self.apply("use-cooled", search_node_id="N0", artifact=path, reason="matching conditions")
        old = (self.root / path).read_bytes()
        new = self.store.invalidate_cooled(path, "needs_revalidation", "v2", "toolchain changed")
        self.assertEqual((self.root / path).read_bytes(), old)
        self.assertEqual(self.store.invalidate_cooled(path, "needs_revalidation", "v2", "toolchain changed"), new)
        for artifact in (path, new):
            with self.subTest(artifact=artifact), self.assertRaises(search.Invalid):
                self.apply("use-cooled", search_node_id="N0", artifact=artifact, reason="reuse")
        self.assertEqual(len(self.store.load()["cooled_inputs"]), 1)
        self.assertEqual(len(json.loads(self.cli("show").stdout)["cooled_warnings"]), 1)

    def test_paths_cannot_escape_project(self):
        for path in ("../outside", str(self.root / "absolute")):
            with self.subTest(path=path), self.assertRaises(search.Invalid):
                self.store.local(path)
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(search.Invalid):
            self.store.local("escape/file")

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
        with self.assertRaisesRegex(search.Invalid, "artifact changed"):
            self.store.load()

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

    def cli(self, *args, script=SCRIPT):
        return subprocess.run([sys.executable, str(script), "--workspace", str(self.root), "--project", "example", "--owner", "session", *args],
                              text=True, capture_output=True)

    def test_read_commands_do_not_write_and_missing_tree_does_not_initialize(self):
        def snapshot():
            return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                    for p in self.root.rglob("*") if p.is_file()}

        before = snapshot()
        for command in ("show", "check", "review"):
            result = self.cli(command)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(), before)
        self.store.path.unlink()
        before = snapshot()
        self.assertNotEqual(self.cli("show").returncode, 0)
        self.assertEqual(snapshot(), before)

    def test_concurrent_writers_keep_both_branches(self):
        processes = []
        for key in ("N1", "N2"):
            operation = {"event_id": key, "op": "add-node", "id": key, "parent": "N0",
                         "hypothesis": key, "scope": "all"}
            file = self.root / f"{key}.json"
            file.write_text(json.dumps(operation))
            processes.append(subprocess.Popen(
                [sys.executable, str(SCRIPT), "--workspace", str(self.root), "--project", "example", "--owner", "session", "apply", str(file)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            _, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, error)
        self.assertEqual(set(self.store.load()["nodes"]), {"N0", "N1", "N2"})

    def test_installed_helper_works_outside_source_repository(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/install_codex_skills.py"),
                                 "--project-dir", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = self.root / ".agents/skills/harness-resume/scripts/searchctl.py"
        result = self.cli("check", script=installed)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"valid": True, "cooled_warnings": []})
        self.assertIn("searchctl.py", (self.root / "AGENTS.md").read_text())

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
        return {"event_id": "batch-event", "op": "develop-batch", "id": "B1",
                "candidate_id": "C0", "inputs": inputs, "members": members}

    def finish_member(self, key, status="completed"):
        task = self.root / f".workspace/example/runs/{key}/TASK.md"
        result = self.write(f".workspace/example/runs/{key}/RESULT.md",
                            task.read_text() + f"- status: {status}\n" + self.acceptance_fields(key))
        return self.apply("accept", id=key, result=result)

    def test_batch_cross_strategy_restart_and_out_of_order_results(self):
        operation = self.batch_fixture()
        self.store.apply(operation)
        before = self.store.path.read_bytes()
        self.store.apply(operation)
        self.assertEqual(before, self.store.path.read_bytes())
        self.store = search.Store(self.root, "example", "session")
        self.assertEqual(self.store.load()["current_node"], "N1")
        self.finish_member("d1")
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.apply("select", id="N0", reason="too early")
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.record_run("extra")
        tree = self.finish_member("d0")
        self.assertEqual(tree["batches"]["B1"]["members"], ["d0", "d1"])
        self.assertTrue(tree["runs"]["d1"]["result"])
        self.apply("select", id="N0", reason="all results retained")

    def test_batch_rejects_shared_worktree_or_task_without_partial_registration(self):
        operation = self.batch_fixture()
        for field in ("repo_ids", "task", "id"):
            bad = deepcopy(operation)
            bad["members"][1][field] = bad["members"][0][field]
            before = self.store.path.read_bytes()
            with self.assertRaises(search.Invalid):
                self.store.apply(bad)
            self.assertEqual(before, self.store.path.read_bytes())
            self.assertNotIn("d0", self.store.load()["runs"])

    def test_batch_checks_analysis_identity_and_comparison_candidate(self):
        operation = self.batch_fixture()
        for analysis in ("missing", "a0"):
            bad = deepcopy(operation)
            bad["members"][1]["analysis_run"] = analysis
            # Keep TASK metadata consistent: the analysis relationship must reject it.
            path = self.root / bad["members"][1]["task"]
            original = path.read_text()
            path.write_text(original.replace("analysis_run: a1", f"analysis_run: {analysis}"))
            with self.assertRaisesRegex(search.Invalid, "analysis run"):
                self.store.apply(bad)
            path.write_text(original)
        self.candidate("C1", node="N1", parent="C0")
        with self.assertRaisesRegex(search.Invalid, "analysis run"):
            self.store.apply({**operation, "candidate_id": "C1"})
        self.assertNotIn("batches", self.store.load())

    def test_batch_rejects_unaccepted_or_failed_analysis(self):
        operation = self.batch_fixture()
        # Existing terminal failure is a valid ledger record, but not a proposal.
        tree = self.store.load()
        path = self.store.local(tree["runs"]["a1"]["result"]["path"])
        path.write_text(path.read_text().replace("status: completed", "status: failed"))
        tree["runs"]["a1"]["result"] = self.store.reference(str(path.relative_to(self.root)))
        search.atomic(self.store.path, tree)
        with self.assertRaisesRegex(search.Invalid, "status must be exactly completed"):
            self.store.apply(operation)
        tree["runs"]["a1"]["result"] = None
        search.atomic(self.store.path, tree)
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.store.apply(operation)

    def test_batch_checks_real_worktree_branch_and_input_hashes(self):
        operation = self.batch_fixture()
        worktree = self.root / ".workspace/example/repos/variant1"
        subprocess.run(["git", "-C", str(worktree), "switch", "-c", "changed"],
                       check=True, capture_output=True)
        before = self.store.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "branch changed"):
            self.store.apply(operation)
        self.assertEqual(before, self.store.path.read_bytes())
        subprocess.run(["git", "-C", str(worktree), "switch", "codex/example/variant1"],
                       check=True, capture_output=True)
        self.store.apply(operation)
        self.write(operation["inputs"]["protocol"], "changed measurement window")
        with self.assertRaisesRegex(search.Invalid, "artifact changed"):
            self.store.load()

    def test_batch_checkpoints_do_not_finish_group_and_new_group_is_blocked(self):
        operation = self.batch_fixture()
        self.store.apply(operation)
        with self.assertRaisesRegex(search.Invalid, "terminal"):
            self.finish_member("d0", "interrupted")
        self.finish_member("d1")
        with self.assertRaisesRegex(search.Invalid, "outstanding"):
            self.store.apply({**operation, "id": "B2", "event_id": "batch2"})
        self.assertIsNone(self.store.load()["runs"]["d0"]["result"])

    def test_old_device_test_record_can_be_received_but_not_created(self):
        text = "- project_id: example\n- run_id: old\n- node: device-test\n- search_node_id: N0\n"
        task = self.write(".workspace/example/runs/old/TASK.md", text)
        with self.assertRaisesRegex(search.Invalid, "legacy only"):
            self.apply("run", id="old", search_node_id="N0", stage="device-test",
                       candidate_id=None, task=task)
        tree = self.store.load()
        tree["runs"]["old"] = {"search_node_id": "N0", "stage": "device-test",
                                "candidate_id": None, "task": self.store.reference(task), "result": None}
        search.atomic(self.store.path, tree)
        result = self.write(".workspace/example/runs/old/RESULT.md", text + "- status: completed\n")
        self.apply("accept", id="old", result=result)

    def test_ordinary_develop_cannot_bypass_batch_isolation(self):
        self.record_run("first", "develop", finish=False)
        with self.assertRaisesRegex(search.Invalid, "must run alone"):
            self.record_run("second", "develop", finish=False)
        with self.assertRaisesRegex(search.Invalid, "must run alone"):
            self.record_run("analysis-while-editing", finish=False)

    def port_fixture(self):
        """Synthetic port evidence; intentionally no optimization-idea diagnosis."""
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
        operation = dict(id="port", search_node_id="N0", stage="port-baseline", candidate_id=None, task=task)
        return operation, result, mapping, data

    def test_port_accepts_correct_measured_slow_initial_candidate_and_freezes_mapping(self):
        operation, result, mapping, _ = self.port_fixture()
        self.apply("run", **operation)
        self.apply("accept", id="port", result=result)
        run = self.store.load()["runs"]["port"]
        self.assertEqual(run["acceptance"]["policy"], "port-acceptance@1")
        self.assertNotIn("regression", run["acceptance"]["evidence"])
        self.apply("candidate", id="Cport", search_node_id="N0", parent_candidate=None,
                   manifest=".workspace/example/runs/port/CANDIDATE.md")
        self.record_run("inspect-port", candidate="Cport")
        with (self.root / mapping).open("a") as stream:
            stream.write("changed mapping")
        with self.assertRaisesRegex(search.Invalid, "evidence changed"):
            self.store.load()

    def test_port_rejects_failed_accuracy_invalid_measurement_and_incomplete_replacement(self):
        operation, result, mapping, data = self.port_fixture()
        self.apply("run", **operation)
        failures = ((result, "accuracy_status: passed", "accuracy_status: failed"),
                    (result, "status: completed", "status: failed"),
                    (result, "performance_status: failed", "performance_status: invalid"),
                    (result, "performance_status: failed", "performance_status: not_run"),
                    (mapping, "status: complete", "status: incomplete"),
                    (mapping, "aicpu_status: replaced", "aicpu_status: unresolved"),
                    (mapping, "candidate_id: Cport", "candidate_id: sibling"),
                    (data, "candidate_id: Cport", "candidate_id: sibling"),
                    (data, "status: ready", "status: partial"))
        for path, old, new in failures:
            with self.subTest(new=new):
                file = self.root / path
                original = file.read_text()
                file.write_text(original.replace(old, new))
                try:
                    with self.assertRaises(search.Invalid):
                        self.apply("accept", id="port", result=result)
                    self.assertIsNone(self.store.load()["runs"]["port"]["result"])
                finally:
                    file.write_text(original)
        self.apply("accept", id="port", result=result)

    def test_port_cannot_change_scope_or_run_with_other_workers(self):
        operation, result, _, _ = self.port_fixture()
        self.record_run("active-analysis", finish=False)
        with self.assertRaisesRegex(search.Invalid, "must run alone"):
            self.apply("run", **operation)
        task = self.root / ".workspace/example/runs/active-analysis/TASK.md"
        done = self.write(".workspace/example/runs/active-analysis/RESULT.md", task.read_text() + "- status: completed\n")
        self.apply("accept", id="active-analysis", result=done)
        self.apply("run", **operation)
        for stage in ("analysis", "develop"):
            with self.assertRaisesRegex(search.Invalid, "must run alone"):
                self.record_run("concurrent-" + stage, stage, finish=False)
        with self.assertRaisesRegex(search.Invalid, "frozen scope"):
            self.apply("accept", id="port", result=result, scope=operation["task"])

    def test_port_requires_validation_and_forbids_aicpu_in_task(self):
        operation, _, _, _ = self.port_fixture()
        file = self.root / operation["task"]
        original = file.read_text()
        for old, new in (("port-and-validate", "code-only"), ("aicpu_policy: forbidden", "aicpu_policy: allowed"),
                         ("performance_required: true", "performance_required: false"),
                         ("diagnostics_required: true", "diagnostics_required: unknown")):
            file.write_text(original.replace(old, new))
            with self.subTest(new=new), self.assertRaises(search.Invalid):
                self.apply("run", **operation)
            self.assertNotIn("port", self.store.load()["runs"])
        file.write_text(original)
        self.apply("run", **operation)

    def test_accuracy_failure_cannot_be_accepted_or_hide_behind_failed_status(self):
        operation = self.batch_fixture()
        self.store.apply(operation)
        self.finish_member("d1")
        task = self.root / ".workspace/example/runs/d0/TASK.md"
        for status, accuracy in (("completed", "failed"), ("failed", "failed"),
                                 ("completed", "invalid"), ("completed", "not_run")):
            with self.subTest(status=status, accuracy=accuracy):
                result = self.write(".workspace/example/runs/d0/RESULT.md", task.read_text() +
                                    f"- status: {status}\n" + self.acceptance_fields("d0", accuracy=accuracy))
                before = self.store.path.read_bytes()
                with self.assertRaises(search.Invalid):
                    self.apply("accept", id="d0", result=result)
                self.assertEqual(self.store.path.read_bytes(), before)
                self.assertTrue(self.store.load()["runs"]["d1"]["result"])
        # A repaired candidate is revalidated in the same node/run, without redispatch.
        tree = self.finish_member("d0")
        self.assertTrue(tree["runs"]["d0"]["acceptance"]["evidence"]["accuracy"])

    def test_performance_regression_needs_idea_evidence_and_freezes_it(self):
        self.record_run("d", "develop", finish=False)
        task = self.root / ".workspace/example/runs/d/TASK.md"
        for performance, cause in (("failed", "implementation"), ("passed", "implementation"),
                                   ("failed", "unknown"), ("failed", "none"),
                                   ("invalid", "idea"), ("not_run", "none")):
            with self.subTest(performance=performance, cause=cause):
                result = self.write(".workspace/example/runs/d/RESULT.md", task.read_text() +
                                    "- status: completed\n" + self.acceptance_fields(
                                        "d", performance=performance, cause=cause))
                with self.assertRaises(search.Invalid):
                    self.apply("accept", id="d", result=result)
                self.assertIsNone(self.store.load()["runs"]["d"]["result"])
        text = task.read_text() + "- status: completed\n" + self.acceptance_fields(
            "d", performance="failed", cause="idea")
        result = self.write(".workspace/example/runs/d/RESULT.md", text)
        self.write(".workspace/example/runs/d/regression.md", "")
        with self.assertRaisesRegex(search.Invalid, "empty develop evidence"):
            self.apply("accept", id="d", result=result)
        self.write(".workspace/example/runs/d/regression.md", "Synthetic idea diagnosis; implementation checked.")
        self.apply("accept", id="d", result=result)
        self.write(".workspace/example/runs/d/regression.md", "Changed explanation")
        with self.assertRaisesRegex(search.Invalid, "evidence changed"):
            self.store.load()

    def test_explicit_limited_modes_do_not_authorize_scope_downgrades(self):
        for mode in ("code-only", "collect-only", "verify-only"):
            key = mode
            task = self.write(f".workspace/example/runs/{key}/TASK.md",
                              f"- project_id: example\n- run_id: {key}\n- node: develop\n"
                              f"- search_node_id: N0\n- operation: {mode}\n- performance_required: false\n")
            self.apply("run", id=key, search_node_id="N0", stage="develop", candidate_id=None, task=task)
            accuracy = "passed" if mode == "verify-only" else "not_run"
            result = self.write(f".workspace/example/runs/{key}/RESULT.md",
                                self.store.local(task).read_text() + "- status: completed\n" +
                                self.acceptance_fields(key, accuracy=accuracy, performance="not_run"))
            self.apply("accept", id=key, result=result)
        self.record_run("full", "develop", finish=False)
        scope = self.write("scope.md", "- project_id: example\n- run_id: full\n"
                           "- operation: code-only\n- performance_required: false\n")
        result = self.write(".workspace/example/runs/full/RESULT.md",
                            "- project_id: example\n- run_id: full\n- node: develop\n"
                            "- search_node_id: N0\n- status: completed\n" +
                            self.acceptance_fields("full", accuracy="not_run", performance="not_run"))
        with self.assertRaisesRegex(search.Invalid, "cannot replace"):
            self.apply("accept", id="full", result=result, scope=scope)
        with self.assertRaisesRegex(search.Invalid, "accuracy cases"):
            self.apply("accept", id="full", result=result)

    def test_legacy_pending_develop_uses_separate_scope_without_editing_task(self):
        text = "- project_id: example\n- run_id: old\n- node: develop\n- search_node_id: N0\n"
        task = self.write(".workspace/example/runs/old/TASK.md", text)
        tree = self.store.load()
        tree["runs"]["old"] = {"search_node_id": "N0", "stage": "develop", "candidate_id": None,
                                "task": self.store.reference(task), "result": None}
        search.atomic(self.store.path, tree)
        scope = self.write(".workspace/example/runs/old/scope.md", text +
                           "- operation: implement-and-validate\n- performance_required: true\n")
        result = self.write(".workspace/example/runs/old/RESULT.md", text + "- status: completed\n" +
                            self.acceptance_fields("old"))
        self.apply("accept", id="old", result=result, scope=scope)
        self.assertEqual(self.store.local(task).read_text(), text)
        self.assertIn("acceptance_scope", self.store.load()["runs"]["old"])

    def test_required_analysis_data_must_be_ready_and_bound_to_run(self):
        task_text = ("- project_id: example\n- run_id: d\n- node: develop\n- search_node_id: N0\n"
                     "- operation: implement-and-validate\n- performance_required: true\n"
                     "- diagnostics_required: true\n")
        task = self.write(".workspace/example/runs/d/TASK.md", task_text)
        self.apply("run", id="d", search_node_id="N0", stage="develop", candidate_id=None, task=task)
        result_text = task_text + "- status: completed\n" + self.acceptance_fields("d")
        result = self.write(".workspace/example/runs/d/RESULT.md", result_text)
        with self.assertRaisesRegex(search.Invalid, "analysis_data"):
            self.apply("accept", id="d", result=result)
        data = ".workspace/example/runs/d/ANALYSIS_DATA.md"
        self.write(result, result_text + f"- analysis_data: {data}\n")
        for status in ("partial", "unavailable", "not_run", "not_applicable"):
            self.write(data, f"- project_id: example\n- run_id: d\n- status: {status}\n")
            with self.assertRaisesRegex(search.Invalid, "not ready"):
                self.apply("accept", id="d", result=result)
        self.write(data, "- project_id: example\n- run_id: sibling\n- status: ready\n")
        with self.assertRaisesRegex(search.Invalid, "run_id"):
            self.apply("accept", id="d", result=result)
        self.write(data, "- project_id: example\n- run_id: d\n- status: ready\n")
        self.apply("accept", id="d", result=result)
        self.assertIn("analysis_data", self.store.load()["runs"]["d"]["acceptance"]["evidence"])
        self.write(data, "- project_id: example\n- run_id: d\n- status: ready\nchanged evidence")
        with self.assertRaisesRegex(search.Invalid, "evidence changed"):
            self.store.load()

    def test_installed_batch_with_isolated_workers_and_shared_analysis(self):
        operation = self.batch_fixture()
        # One accepted analysis may provide two independent plans in the same strategy.
        member = operation["members"][1]
        member.update(search_node_id="N0", analysis_run="a0")
        task = self.root / member["task"]
        task.write_text(task.read_text().replace("search_node_id: N1", "search_node_id: N0")
                        .replace("analysis_run: a1", "analysis_run: a0"))
        installed = subprocess.run([sys.executable, str(ROOT / "scripts/install_codex_skills.py"),
                                    "--workspace", str(self.root)], capture_output=True, text=True)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        operation_file = self.root / "batch.json"
        operation_file.write_text(json.dumps(operation))
        script = self.root / ".agents/skills/harness-resume/scripts/searchctl.py"
        result = self.cli("apply", str(operation_file), script=script)
        self.assertEqual(result.returncode, 0, result.stderr)
        # Synthetic CPU workers exercise directory isolation; these are not NPU results.
        processes = []
        for index in range(2):
            worktree = self.root / f".workspace/example/repos/variant{index}"
            code = ("from pathlib import Path; import sys; "
                    "p=Path('candidate.py'); p.write_text('value = '+sys.argv[1]); "
                    "ns={}; exec(p.read_text(), ns); assert ns['value'] == int(sys.argv[1])")
            processes.append(subprocess.Popen([sys.executable, "-c", code, str(index)],
                                               cwd=worktree, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True))
        for process in processes:
            _, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, error)
        for index in range(2):
            path = self.root / f".workspace/example/repos/variant{index}/candidate.py"
            self.assertEqual(path.read_text(), f"value = {index}")
        self.assertFalse((self.root / "source/candidate.py").exists())
        self.finish_member("d1")
        self.finish_member("d0")
        self.assertEqual(self.cli("check", script=script).returncode, 0)


if __name__ == "__main__":
    unittest.main()
