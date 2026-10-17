"""Build the cross_condition knowledge graph SQLite from run artifacts.

Numbers are parsed ONLY from files under experiments/cross_condition/runs/:
  config.json, git_info.json, metrics_epoch.csv,
  evaluation_results/master_benchmark_summary.json,
  evaluation_results/master_benchmark_metrics.csv,
  diagnose_readouts_v{5,4,3,}/readout_diagnostics.csv (v5 preferred; adds P1u_trim_s2),
  evaluation_results/diagnose_readouts/readout_diagnostics.csv (legacy v2).

Causal layer (hypotheses, failures, invariants, verdicts, edges) is loaded
from seed_causal.json (human-curated, rationale+source required).

Usage (CPU-only, rerun any time; exp53/54 get picked up once finished):
  C:\\Users\\tranl\\miniconda3\\envs\\ndt\\python.exe ^
    -m src.PECT_JEPA.cross_condition.knowledge.build ^
    --db src\\PECT_JEPA\\cross_condition\\knowledge\\graph.sqlite
"""

import argparse
import csv
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from src.PECT_JEPA.cross_condition.knowledge.schema import DDL  # noqa: E402


def upsert_node(cur, nid, ntype, label, attrs=None, source="", commit=""):
    cur.execute(
        "INSERT INTO nodes(id,type,label,attrs,source,commit_hash) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET type=excluded.type, label=excluded.label, "
        "attrs=excluded.attrs, source=excluded.source, commit_hash=excluded.commit_hash",
        (nid, ntype, label, json.dumps(attrs or {}, ensure_ascii=False), source, commit),
    )


def add_edge(cur, src, dst, rel, rationale="", source="", author="curator", commit=""):
    cur.execute(
        "INSERT INTO edges(src,dst,rel,rationale,source,author,commit_hash) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(src,dst,rel) DO UPDATE SET rationale=excluded.rationale, "
        "source=excluded.source, author=excluded.author, commit_hash=excluded.commit_hash",
        (src, dst, rel, rationale, source, author, commit),
    )


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fnum(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def parse_training_metrics(run_dir):
    """metrics_epoch.csv -> {epochs, max_val_auc(+epoch), final_rank}."""
    p = os.path.join(run_dir, "metrics_epoch.csv")
    if not os.path.exists(p):
        return {}
    epochs, best_auc, best_ep, final_rank = 0, None, None, None
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                epochs += 1
                a = fnum(row.get("val_auc"))
                if a is not None and (best_auc is None or a > best_auc):
                    best_auc, best_ep = a, fnum(row.get("epoch"))
                r = fnum(row.get("effective_rank"))
                if r is not None:
                    final_rank = r
    except OSError:
        return {}
    return {"epochs": epochs, "telemetry_val_auc_max": best_auc,
            "telemetry_val_auc_epoch": best_ep, "final_effective_rank": final_rank}


def parse_suite_metrics(run_dir):
    """master_benchmark_summary.json + per-file metrics.csv -> metric records.

    Returns list of (metric_id_suffix, name, value, split, slice, readout).
    """
    recs = []
    s = read_json(os.path.join(run_dir, "evaluation_results", "master_benchmark_summary.json"))
    if s:
        for split, blk in (s.get("by_split") or {}).items():
            for key in ("linear_probe_auc_mean", "linear_probe_ap_mean",
                        "mahalanobis_auc_mean", "cnr_mean",
                        "segmentation_iou_mean", "segmentation_dice_mean"):
                v = blk.get(key)
                if v is not None:
                    recs.append((f"suite:{split}:{key}", key, float(v), split, "all", "suite"))
        for sensor, blk in (s.get("by_sensor") or {}).items():
            v = blk.get("linear_probe_auc_mean")
            if v is not None:
                recs.append((f"suite:all:sensor:{sensor}:lp_auc", "linear_probe_auc_mean",
                             float(v), "all", sensor, "suite"))
    mp = os.path.join(run_dir, "evaluation_results", "master_benchmark_metrics.csv")
    if os.path.exists(mp):
        try:
            with open(mp, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            test = [r for r in rows if r.get("split") == "test"]
            sensors = sorted({r.get("sensor", "?") for r in test})
            for sensor in sensors:
                vals = [fnum(r.get("linear_probe_auc")) for r in test if r.get("sensor") == sensor]
                m = mean(vals)
                if m is not None:
                    recs.append((f"suite:test:sensor:{sensor}:lp_auc", "linear_probe_auc_mean",
                                 m, "test", sensor, "suite"))
        except OSError:
            pass
    return recs


READOUT_CANDIDATES = [
    ("diagnose_readouts_v6", "v6"),
    ("diagnose_readouts_v5", "v5"),
    ("diagnose_readouts_v4", "v4"),
    ("diagnose_readouts_v3", "v3"),
    ("evaluation_results/diagnose_readouts", "v2"),
]


def parse_readout_metrics(run_dir):
    """Best-available readout_diagnostics.csv -> mean AUC/AP per split (+test by sensor)."""
    for sub, ver in READOUT_CANDIDATES:
        p = os.path.join(run_dir, sub, "readout_diagnostics.csv")
        if os.path.exists(p):
            path, version = p, ver
            break
    else:
        return [], None, None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return [], None, None
    if not rows:
        return [], None, None
    cols = [c for c in rows[0].keys()
            if c.startswith("auc_") or c.startswith("ap_") or c in ("iou_otsu", "f1_otsu")]
    recs = []
    for split in ("train", "val", "test"):
        grp = [r for r in rows if r.get("split") == split] or rows
        for c in cols:
            m = mean([fnum(r.get(c)) for r in grp])
            if m is not None:
                recs.append((f"readout:{version}:{split}:{c}", c, m, split, "all", version))
    key_cols = [c for c in ("auc_raw", "auc_p1u", "auc_p1u_s5", "auc_p1u_maha", "auc_p1u_s2",
                            "auc_p1u_trim_s2", "auc_mlp_p1u_s2", "auc_p2", "auc_p3", "maha_auc")
                if c in (rows[0].keys() or [])]
    for sensor in sorted({r.get("sensor", "?") for r in rows if r.get("split") == "test"}):
        for c in key_cols:
            m = mean([fnum(r.get(c)) for r in rows
                      if r.get("split") == "test" and r.get("sensor") == sensor])
            if m is not None:
                recs.append((f"readout:{version}:test:sensor:{sensor}:{c}", c, m,
                             "test", sensor, version))
    return recs, version, path


def build(runs_dir, db_path, seed_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(DDL)

    seed = read_json(seed_path) or {}
    for h in seed.get("hypotheses", []):
        upsert_node(cur, h["id"], "Hypothesis", h["label"],
                    {"status": h.get("status", "?")}, source="seed_causal.json")
    for fm in seed.get("failures", []):
        upsert_node(cur, fm["id"], "FailureMechanism", fm["label"], source="seed_causal.json")
    for inv in seed.get("invariants", []):
        upsert_node(cur, inv["id"], "Invariant", inv["label"],
                    source="seed_causal.json:invariants")
    for r in seed.get("readouts", []):
        upsert_node(cur, r["id"], "Readout", r["label"],
                    {"status": r.get("status", "?")}, source="seed_causal.json")
    for c in seed.get("components", []):
        upsert_node(cur, c["id"], "Component", c["label"], source="seed_causal.json")

    run_dirs = sorted(d for d in os.listdir(runs_dir)
                      if os.path.isdir(os.path.join(runs_dir, d)))
    n_metrics = 0
    report = []
    for run in run_dirs:
        rdir = os.path.join(runs_dir, run)
        cfg = read_json(os.path.join(rdir, "config.json")) or {}
        git = read_json(os.path.join(rdir, "git_info.json")) or {}
        commit = git.get("commit", "")
        ov = (seed.get("run_overrides") or {}).get(run, {})
        has_eval = os.path.exists(
            os.path.join(rdir, "evaluation_results", "master_benchmark_summary.json"))
        status = ov.get("status", "EVALUATED" if has_eval else "NO_EVAL_YET")
        attrs = {"status": status, "telemetry": ov.get("telemetry", "?"),
                 "verdict": ov.get("verdict", ""),
                 "tokenizer_type": cfg.get("tokenizer_type", "?"),
                 "inv_weight": cfg.get("inv_weight", "?"),
                 "inv_cosine": cfg.get("inv_cosine", "?"),
                 "inv_predictor": cfg.get("inv_predictor", "?"),
                 "inv_oracle_mask": cfg.get("inv_oracle_mask", "?"),
                 "inv_start_epoch": cfg.get("inv_start_epoch", "?"),
                 "xcov_weight": cfg.get("xcov_weight", "?"),
                 "config": cfg}
        attrs.update(parse_training_metrics(rdir))
        upsert_node(cur, run, "Run", cfg.get("exp_name", run), attrs,
                    source=os.path.join(rdir, "config.json"), commit=commit)
        cfg_node = f"artifact:{run}:config.json"
        upsert_node(cur, cfg_node, "Artifact", f"{run}/config.json",
                    {"kind": "config"}, source=os.path.join(rdir, "config.json"), commit=commit)
        add_edge(cur, run, cfg_node, "MEASURED_ON", "config snapshot", cfg_node, commit=commit)

        for tok in ("tokenizer_type",):
            if cfg.get(tok):
                cid = f"component:tokenizer={cfg[tok]}"
                upsert_node(cur, cid, "Component", f"{tok}={cfg[tok]}",
                            {"from_config": True}, source=os.path.join(rdir, "config.json"))
                add_edge(cur, run, cid, "CONFIGURED_WITH", f"{tok}={cfg[tok]}",
                         os.path.join(rdir, "config.json"))

        suite_recs = parse_suite_metrics(rdir)
        ro_recs, ro_ver, ro_path = parse_readout_metrics(rdir)
        for suffix, name, value, split, slc, ro in suite_recs + ro_recs:
            mid = f"metric:{run}:{suffix}"
            art = ("evaluation_results/master_benchmark_summary.json"
                   if suffix.startswith("suite:") and ":sensor:" not in suffix
                   else ("evaluation_results/master_benchmark_metrics.csv"
                         if ":sensor:" in suffix and ro == "suite" else ro_path))
            aid = f"artifact:{run}:{os.path.basename(art)}"
            upsert_node(cur, aid, "Artifact", f"{run}/{os.path.basename(art)}",
                        {"kind": "eval"}, source=os.path.join(rdir, art), commit=commit)
            upsert_node(cur, mid, "Metric", f"{run} {split}/{slc} {name}={value:.4f}",
                        {"value": value, "metric": name, "split": split,
                         "slice": slc, "readout": ro}, source=os.path.join(rdir, art),
                        commit=commit)
            add_edge(cur, mid, aid, "MEASURED_ON", "parsed value", art, commit=commit)
            n_metrics += 1
        if ro_ver:
            add_edge(cur, run, "R-P0", "EVALUATED_WITH", f"readout matrix {ro_ver}",
                     ro_path or "")

        t = next((v for sfx, n, v, sp, sl, ro in suite_recs
                  if sfx == "suite:test:linear_probe_auc_mean"), None)
        p1u = next((v for sfx, n, v, sp, sl, ro in ro_recs
                    if n == "auc_p1u_s2" and sp == "test" and sl == "all"), None)
        report.append((run, status, t, p1u, ro_ver))

    for e in seed.get("edges", []):
        add_edge(cur, e["src"], e["dst"], e["rel"], e.get("rationale", ""),
                 e.get("source", ""), commit="")
    con.commit()

    print(f"runs={len(run_dirs)} metrics={n_metrics} "
          f"nodes={cur.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]} "
          f"edges={cur.execute('SELECT COUNT(*) FROM edges').fetchone()[0]}")
    print(f"{'run':45s} {'status':24s} {'testLP':>7s} {'testP1u_s2':>10s} readout")
    for run, status, t, p1u, rover in report:
        ts = f"{t:.4f}" if t is not None else "    n/a"
        ps = f"{p1u:.4f}" if p1u is not None else "      n/a"
        print(f"{run:45s} {status:24s} {ts:>7s} {ps:>10s} {rover or '-'}")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join("experiments", "cross_condition", "runs"))
    ap.add_argument("--db", default=os.path.join("src", "PECT_JEPA", "cross_condition",
                                                 "knowledge", "graph.sqlite"))
    ap.add_argument("--seed", default=os.path.join("src", "PECT_JEPA", "cross_condition",
                                                   "knowledge", "seed_causal.json"))
    a = ap.parse_args()
    build(a.runs_dir, a.db, a.seed)
