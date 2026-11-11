# Search nodes and hot/cooled knowledge

Search nodes preserve optimization alternatives that a flat execution log cannot express. Each node owns a direction and scope, and may contain many analysis → develop（实现与验证） iterations. Candidate lineage and workflow stages remain independent of the strategy tree.

```text
External cooled knowledge
        |  retrieve applicable, versioned evidence for any node
        v
Project: operator or operator group
N0 Initial implementation                   [Analyze -> Develop+Validate] x n
├── N1 Memory optimization                  [Analyze -> Develop+Validate] x n
│   ├── N3 Layout                           [Analyze -> Develop+Validate] x n
│   │   └── N5 Convert during load          [Analyze -> Develop+Validate] x n
│   └── N4 Increase reuse                   [Analyze -> Develop+Validate] x n
└── N2 Compute organization                 [Analyze -> Develop+Validate] x n
    └── N6 Fuse internal stages             [Analyze -> Develop+Validate] x n
        └── N7 Pipeline changes             [Analyze -> Develop+Validate] x n

Hot H1 <-- N3                 one node
Hot H2 <-- N1, N3, N5         multiple related nodes
Hot H3 <-- N4, N7             evidence across branches
        |
        v
Delivery: consolidate -> select -> human review -> versioned cooled
          unselected / unapproved entries remain hot
```

The stdlib `searchctl.py` helper persists a project-scoped tree, stable run/candidate references, immutable hot revisions, and reviewed cooled packages. Locked atomic writes and idempotent event IDs make metadata updates recoverable without repeating experiments. Frozen artifact hashes detect drift. Hot evidence is captured as immutable content-addressed snapshots within each project so live execution notes can evolve without invalidating previous findings. New records include `project_id` and `search_node_id`, and mutations check project ownership and lifecycle. Existing legacy trees remain explicitly readable with `--legacy`; they are not silently migrated. Adopt the tree at a safe boundary, never reconstruct unknown history.

Cooled knowledge is an external input with explicit applicability, limitations, evidence summaries and review provenance. Promotion is a delivery-time human decision, separate from publication. Invalidation creates an immutable successor; past evidence remains inspectable. This is auditable local bookkeeping, not an authentication service or a correctness/performance gate.

See the distributable [workflow, schemas and CLI reference](../../assets/skills/harness-resume/references/search-knowledge.md). No MCTS, vector database, automatic execution or remote publishing is introduced.

New projects use the [search@2 dependency queue](../../assets/skills/harness-resume/references/task-queue.md). Each analysis/develop run binds explicit dependencies, frozen inputs and its own workspace. Accepted candidates may feed analysis while unrelated develop runs continue; writers and jobs must stop before their workspace lease is released. `current_node` is navigation. Comparison groups freeze comparable inputs without adding an implicit wait barrier. Existing search@1 runs retain the [develop-batch rules](../../assets/skills/harness-resume/references/parallel-develop.md) until a safe upgrade boundary. The host dispatches agents and coordinates device access.
