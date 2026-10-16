"""Knowledge-graph schema for cross_condition SSL research.

Phase 1: deterministic, file-grounded. Numbers are ONLY parsed from run
artifacts (config.json, metrics_epoch.csv, master_benchmark_*.json/csv,
readout_diagnostics.csv, git_info.json). Causal edges (TESTS,
REJECTED_BECAUSE, ...) are human-curated in seed_causal.json with explicit
rationale + source, never inferred by an LLM.

Storage: SQLite (stdlib only) + NetworkX-compatible edge list export.
No Neo4j, no new dependencies.
"""

NODE_TYPES = (
    "Run",            # one concrete run dir, e.g. exp49_inv_xcov_20261014_021809
    "Hypothesis",     # testable claim, e.g. H1 blind-inv helps OOD
    "Component",      # tokenizer / loss / head / schedule knob setting
    "Metric",         # one number: run + split + slice + readout + source file
    "FailureMechanism",  # root-cause mechanism, e.g. FM1 blind-inv washout
    "Invariant",      # constraint from GEMINI.md
    "Artifact",       # code file or run output file (+ commit hash)
    "Readout",        # eval-side probe recipe, e.g. P1u_s2+LR, Otsu
)

EDGE_RELS = (
    "TESTS",               # Run -> Hypothesis
    "MODIFIES",            # Run -> Component (this run sets this knob)
    "CONFIGURED_WITH",     # Run -> Component (full config snapshot link)
    "REJECTED_BECAUSE",    # Hypothesis|Run|Readout -> FailureMechanism
    "ACCEPTED_BECAUSE",    # Hypothesis|Readout -> Metric (supporting evidence)
    "CAUSED_BY",           # FailureMechanism -> Component
    "FIXES",               # Component|Run -> FailureMechanism
    "VIOLATES_INVARIANT",  # Hypothesis|Run -> Invariant
    "SUPERSEDES",          # Run -> Run | Readout -> Readout
    "MEASURED_ON",         # Metric -> Artifact
    "EVALUATED_WITH",      # Run -> Readout
)

DDL = """
CREATE TABLE IF NOT EXISTS nodes(
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  label TEXT NOT NULL,
  attrs TEXT DEFAULT '{}',
  source TEXT DEFAULT '',
  commit_hash TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS edges(
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  rel TEXT NOT NULL,
  rationale TEXT DEFAULT '',
  source TEXT DEFAULT '',
  author TEXT DEFAULT 'curator',
  commit_hash TEXT DEFAULT '',
  PRIMARY KEY(src, dst, rel)
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel);
"""
