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
        task = self.write(f".workspace/example/runs/{key}/TASK.md", text)
        self.apply("run", id=key, search_node_id=node, stage=stage, candidate_id=candidate, task=task)
        if finish:
            result = self.write(f".workspace/example/runs/{key}/RESULT.md", text + "- status: completed\n")
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
            for stage in ("analysis", "develop", "device-test"):
                self.record_run(f"r{iteration}-{stage}", stage, candidate="C0")
        self.candidate("C1", parent="C0")
        self.add_node()
        self.add_node("N2", "N1")
        self.apply("select", id="N2", reason="independent layout strategy")
        self.candidate("C2", node="N2", parent="C0")
        self.record_run("branch-test", "device-test", "N2", "C2")
        self.apply("select", id="N0", reason="revisit initial hypothesis")
        self.record_run("retest", "device-test", candidate="C1")
        tree = search.Store(self.root, "example").load()
        self.assertEqual(len(tree["nodes"]), 3)
        self.assertEqual(len(tree["runs"]), 8)
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
        self.assertEqual(self.store.review()["candidates"][0]["hot"]["revision"], 2)

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
        with self.assertRaisesRegex(search.Invalid, "conflict"):
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


if __name__ == "__main__":
    unittest.main()
