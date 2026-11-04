# Search nodes and hot/cooled knowledge

Search nodes preserve optimization alternatives that a flat execution log cannot express. Each node owns a direction and scope, and may contain many analysis → develop → device-test iterations. Candidate lineage and workflow stages remain independent of the strategy tree.

```text
External cooled knowledge
        |  retrieve applicable, versioned evidence for any node
        v
Single operator
N0 Initial implementation                   [Analyze -> Develop -> Test] x n
├── N1 Memory optimization                  [Analyze -> Develop -> Test] x n
│   ├── N3 Layout                           [Analyze -> Develop -> Test] x n
│   │   └── N5 Convert during load          [Analyze -> Develop -> Test] x n
│   └── N4 Increase reuse                   [Analyze -> Develop -> Test] x n
└── N2 Compute organization                 [Analyze -> Develop -> Test] x n
    └── N6 Fuse internal stages             [Analyze -> Develop -> Test] x n
        └── N7 Pipeline changes             [Analyze -> Develop -> Test] x n

Hot H1 <-- N3                 one node
Hot H2 <-- N1, N3, N5         multiple related nodes
Hot H3 <-- N4, N7             evidence across branches
        |
        v
Delivery: consolidate -> select -> human review -> versioned cooled
          unselected / unapproved entries remain hot
```

The stdlib `searchctl.py` helper persists a single-operator tree, stable run/candidate references, immutable hot revisions, and reviewed cooled packages. Locked atomic writes and idempotent event IDs make metadata updates recoverable without repeating experiments. Frozen artifact hashes detect drift. Hot evidence is captured as immutable content-addressed snapshots so live execution notes can evolve without invalidating previous findings. New records include `search_node_id`; existing v1 workflow contracts remain readable without migration. Adopt the tree at a safe boundary, never reconstruct unknown history.

Cooled knowledge is an external input with explicit applicability, limitations, evidence summaries and review provenance. Promotion is a delivery-time human decision, separate from publication. Invalidation creates an immutable successor; past evidence remains inspectable. This is auditable local bookkeeping, not an authentication service or a correctness/performance gate.

See the distributable [workflow, schemas and CLI reference](../../assets/skills/harness-resume/references/search-knowledge.md). No MCTS, vector database, automatic execution or remote publishing is introduced.
