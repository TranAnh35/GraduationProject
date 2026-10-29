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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
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
    summary_path = os.path.join(run_dir, "evaluation_results", "master_benchmark_summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(run_dir, "master_benchmark_summary.json")
    s = read_json(summary_path)
    if s:
        # 1. Legacy suite format (by_split, by_sensor)
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

        # 2. Protocol benchmark format (representation_performance, by_*_test, paired_deltas)
        reps = s.get("representation_performance") or s.get("representations") or {}
        for rep_name, rdata in reps.items():
            if isinstance(rdata, dict):
                # Flat or nested structure
                for k, v in rdata.items():
                    if isinstance(v, dict):
                        for split, sdata in v.items():
                            if isinstance(sdata, dict):
                                for mname, mval in sdata.items():
                                    val = fnum(mval)
                                    if val is not None:
                                        sfx = f"proto:{rep_name}:{k}:{split}:{mname}"
                                        recs.append((sfx, mname, val, split, "all", f"{rep_name}_{k}"))
                    else:
                        val = fnum(v)
                        if val is not None:
                            sfx = f"proto:{rep_name}:test:{k}"
                            recs.append((sfx, k, val, "test", "all", rep_name))

        # Per-axis breakdowns
        for axis_key in ("sensor", "waveform", "liftoff", "specimen"):
            axis_data = s.get(f"by_{axis_key}_test") or s.get(f"by_{axis_key}") or {}
            for level, ldata in axis_data.items():
                if isinstance(ldata, dict):
                    for mname, mval in ldata.items():
                        val = fnum(mval)
                        if val is not None:
                            sfx = f"proto:axis:{axis_key}:{level}:{mname}"
                            recs.append((sfx, mname, val, "test", level, f"axis_{axis_key}"))

        per_axis = s.get("per_axis_breakdown") or {}
        for axis_name, axis_data in per_axis.items():
            axis_key = axis_name.replace("by_", "").replace("_test", "")
            for rep_name, rep_axis in (axis_data or {}).items():
                for level, ldata in (rep_axis or {}).items():
                    if isinstance(ldata, dict):
                        for mname, mval in ldata.items():
                            val = fnum(mval)
                            if val is not None:
                                sfx = f"proto:{rep_name}:axis:{axis_key}:{level}:{mname}"
                                recs.append((sfx, mname, val, "test", level, f"{rep_name}_axis"))

        paired = s.get("paired_deltas") or s.get("paired_comparisons") or {}
        for comp_name, cdata in paired.items():
            if isinstance(cdata, dict):
                for mname, mval in cdata.items():
                    val = fnum(mval)
                    if val is not None:
                        sfx = f"proto:paired:{comp_name}:{mname}"
                        recs.append((sfx, mname, val, "test", "all", "paired"))

    mp = os.path.join(run_dir, "evaluation_results", "master_benchmark_metrics.csv")
    if not os.path.exists(mp):
        mp = os.path.join(run_dir, "master_benchmark_metrics.csv")
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


def parse_wo_benchmark_metrics(run_dir, run_id):
    """Parses WO master_benchmark_metrics.csv containing per-scan representation records.
    Returns list of (metric_id_suffix, name, value, split, slice, readout, path, attrs).
    """
    candidates = [
        os.path.join(run_dir, "master_benchmark_metrics.csv"),
        os.path.join(run_dir, "benchmarks", "master_benchmark_metrics.csv"),
        os.path.join(run_dir, "evaluation_results", "master_benchmark_metrics.csv"),
    ]
    csv_path = None
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break
    if not csv_path:
        return []

    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if "representation" not in fieldnames or "raw_lr_auc" not in fieldnames:
                return []  # Not a WO schema CSV
            rows = list(reader)
    except OSError:
        return []

    recs = []
    metric_cols = [
        "raw_lr_auc", "raw_lr_ap", "whitened_lr_auc", "whitened_lr_ap",
        "medcenter_lr_auc", "medcenter_lr_ap", "unsup_maha_auc", "unsup_maha_ap", "unsup_maha_cnr",
        "operational_recall_at_val_tau", "operational_fpr_at_val_tau", "operational_precision_at_val_tau",
        "operational_iou_at_val_tau", "roc_recall_at_fpr1_whitened", "roc_recall_at_fpr5_whitened",
        "cad_sizing_mae_mm", "cad_sizing_r2", "source_const_mean_mae_mm", "source_const_mean_r2",
        "source_const_median_mae_mm", "source_const_median_r2",
        "source_const_pixel_mean_mae_mm", "source_const_pixel_median_mae_mm",
        "source_const_object_mean_mae_mm", "source_const_object_median_mae_mm",
    ]

    for r in rows:
        rep = r.get("representation", "unknown")
        s = r.get("seed", "0")
        fn = r.get("file", "unknown")
        specimen = r.get("specimen", "all")
        sensor = r.get("sensor", "all")
        waveform = r.get("waveform", "all")
        liftoff = r.get("liftoff", "all")

        for col in metric_cols:
            val = fnum(r.get(col))
            if val is not None:
                sfx = f"wo_scan:{rep}:s{s}:{fn}:{col}"
                extra_attrs = {
                    "file": fn,
                    "seed": int(s) if s.isdigit() else s,
                    "representation": rep,
                    "specimen": specimen,
                    "sensor": sensor,
                    "waveform": waveform,
                    "liftoff": liftoff,
                    "metric_name": col,
                }
                recs.append((sfx, col, val, "test", sensor, rep, csv_path, extra_attrs))

    return recs


def parse_audit_metrics(run_dir):
    """audit_results.json -> list of (metric_id_suffix, name, value, split, slice, readout, path)."""
    candidates = [
        os.path.join(run_dir, "audit_results.json"),
        os.path.join(run_dir, "audits", "audit_results.json"),
        os.path.join(run_dir, "..", "audits", "audit_results.json"),
    ]
    audit_path = None
    for c in candidates:
        if os.path.exists(c):
            audit_path = c
            break
    if not audit_path:
        return []
    s = read_json(audit_path) or {}
    audits = s.get("audits", {})
    recs = []

    # R2a
    r2a = audits.get("R2a_mask_dependence", {})
    if r2a:
        for k in ("amplitude_dependent_masks", "sign_dependent_masks",
                  "mean_amplitude_delta_center_hidden", "mean_sign_delta_center_hidden"):
            v = fnum(r2a.get(k))
            if v is not None:
                recs.append((f"audit:r2a:{k}", k, v, "test", "all", "audit_r2a", audit_path))

    # R2b
    r2b = audits.get("R2b_center_vs_context", {})
    if r2b:
        for k in ("center_only_auc_mean", "context_only_auc_mean", "concat_auc_mean",
                  "delta_concat_minus_center", "delta_concat_minus_context",
                  "concat_vs_center_win_rate", "concat_vs_context_win_rate"):
            v = fnum(r2b.get(k))
            if v is not None:
                recs.append((f"audit:r2b:{k}", k, v, "test", "all", "audit_r2b", audit_path))

    # R2c
    r2c = audits.get("R2c_condition_token", {})
    if r2c:
        tpl = r2c.get("target_prediction_loss", {})
        for k in ("matched_condition_loss", "zero_condition_loss", "bypass_none_loss",
                  "donor_same_specimen_loss_mean", "donor_diff_specimen_loss_mean"):
            v = fnum(tpl.get(k))
            if v is not None:
                recs.append((f"audit:r2c:target_loss:{k}", k, v, "test", "all", "audit_r2c", audit_path))
        ord_data = r2c.get("output_representation_distance", {})
        for k in ("distance_matched_vs_zero", "distance_matched_vs_bypass",
                  "distance_matched_vs_donor_same_specimen", "distance_matched_vs_donor_diff_specimen"):
            v = fnum(ord_data.get(k))
            if v is not None:
                recs.append((f"audit:r2c:rep_dist:{k}", k, v, "test", "all", "audit_r2c", audit_path))

    # R2d
    r2d = audits.get("R2d_spatial_shortcut_and_cad_prior", {})
    if r2d:
        for k in ("spatial_variance_ratio", "variance_across_16_masks_on_constant_patch",
                  "oracle_cad_lattice_auc"):
            v = fnum(r2d.get(k))
            if v is not None:
                recs.append((f"audit:r2d:{k}", k, v, "test", "all", "audit_r2d", audit_path))

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
    if version == "v2":
        if "auc_p1u_s5" in rows[0]:
            version = "v6"
        elif "auc_p1u_trim_s2" in rows[0]:
            version = "v5"
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


def resolve_repo_path(p, must_exist=False):
    if not p:
        return p
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    cand = os.path.join(ROOT_DIR, p)
    if os.path.exists(cand):
        return os.path.abspath(cand)
    p_fwd = p.replace("\\", "/")
    marker = "src/PECT_JEPA/cross_condition/"
    if p_fwd.startswith(marker):
        rel = p_fwd[len(marker):]
        if os.path.exists(rel):
            return os.path.abspath(rel)
    return os.path.abspath(cand) if not must_exist else (os.path.abspath(cand) if os.path.exists(cand) else os.path.abspath(p))


def ingest_campaigns(cur):
    """Ingest locked full campaigns; smoke results never enter research evidence.

    Numeric sources remain master_benchmark_*.json/csv. Verdicts are deliberately
    left for review, never inferred from completion or a positive mean delta.
    """
    campaign_root = os.path.join(os.path.dirname(__file__), "..", "experiments", "campaigns")
    if not os.path.isdir(campaign_root):
        return 0
    count = 0
    for name in sorted(os.listdir(campaign_root)):
        directory = os.path.join(campaign_root, name)
        plan = read_json(os.path.join(directory, "plan.json"))
        if not plan or plan.get("profile") != "full" or name.startswith("_"):
            continue
        git = read_json(os.path.join(directory, "git_info.json")) or {}
        rid = f"campaign:{name}"
        upsert_node(cur, rid, "Run", name, {"status": "CAMPAIGN_NEEDS_REVIEW",
            "fingerprint": plan.get("fingerprint"), "config": plan.get("settings", {})},
            source=os.path.join(directory, "plan.json"), commit=git.get("commit", ""))
        for job in plan.get("jobs", []):
            if job.get("kind") != "evaluate":
                continue
            jdir = os.path.join(directory, "jobs", job["id"])
            result = read_json(os.path.join(jdir, "result.json"))
            summary_path = os.path.join(jdir, "master_benchmark_summary.json")
            summary = read_json(summary_path)
            if not result or not result.get("complete") or not summary:
                continue
            aid = f"artifact:{rid}:{job['id']}"
            upsert_node(cur, aid, "Artifact", job["id"], job, source=summary_path,
                        commit=git.get("commit", ""))
            for metric in ("macro_ap", "macro_auc"):
                value = fnum(summary.get(metric))
                if value is None:
                    continue
                mid = f"metric:{rid}:{job['id']}:{metric}"
                upsert_node(cur, mid, "Metric", f"{job['id']} {metric}={value:.4f}",
                    {"value": value, "metric": metric, "split": "test", "slice": job["scenario"],
                     "readout": job["arm"], "representation": job["representation"], "seed": job["seed"]},
                    source=summary_path, commit=git.get("commit", ""))
                add_edge(cur, mid, aid, "MEASURED_ON", "parsed campaign artifact", summary_path)
                add_edge(cur, rid, mid, "EVALUATED_WITH", "campaign test score; verdict pending", summary_path)
                count += 1
    return count


def build(runs_dir, db_path, seed_path, extra_dirs=None):
    runs_dir = resolve_repo_path(runs_dir, must_exist=False)
    db_path = resolve_repo_path(db_path, must_exist=False)
    seed_path = resolve_repo_path(seed_path, must_exist=False)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
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

    run_dirs = []
    if os.path.isdir(runs_dir):
        run_dirs = [(d, os.path.join(runs_dir, d)) for d in sorted(os.listdir(runs_dir))
                    if os.path.isdir(os.path.join(runs_dir, d))]

    # Discovery of WO benchmark directories
    IGNORED_SUBDIRS = {"readouts", "feature_cache", "__pycache__", "cache", "plots"}
    if extra_dirs:
        seen_dirs = set()
        for ed in extra_dirs:
            if not os.path.isabs(ed):
                cand = os.path.join(ROOT_DIR, ed)
                if os.path.isdir(cand):
                    ed = cand
            if not os.path.isdir(ed):
                continue
            norm_ed = os.path.normpath(os.path.abspath(ed))
            if norm_ed in seen_dirs:
                continue
            seen_dirs.add(norm_ed)
            wo_prefix = os.path.basename(norm_ed)

            has_summary = (os.path.exists(os.path.join(ed, "master_benchmark_summary.json"))
                           or os.path.exists(os.path.join(ed, "master_benchmark_metrics.csv"))
                           or os.path.exists(os.path.join(ed, "audit_results.json")))
            if has_summary:
                run_dirs.append((wo_prefix, ed))
                continue

            for d in sorted(os.listdir(ed)):
                if d in IGNORED_SUBDIRS:
                    continue
                sub = os.path.join(ed, d)
                if os.path.isdir(sub):
                    sub_norm = os.path.normpath(os.path.abspath(sub))
                    if sub_norm not in seen_dirs:
                        seen_dirs.add(sub_norm)
                        run_dirs.append((f"{wo_prefix}:{d}", sub))

    n_metrics = 0
    report = []
    for run, rdir in run_dirs:
        cfg = (read_json(os.path.join(rdir, "config.json"))
               or read_json(os.path.join(rdir, "config_resume.json"))
               or read_json(os.path.join(rdir, "protocol_manifest.json"))
               or {})
        git = read_json(os.path.join(rdir, "git_info.json")) or {}
        commit = git.get("commit", "")
        ov = (seed.get("run_overrides") or {}).get(run, {})
        has_eval = (os.path.exists(os.path.join(rdir, "evaluation_results", "master_benchmark_summary.json"))
                    or os.path.exists(os.path.join(rdir, "master_benchmark_summary.json")))
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
        audit_recs = parse_audit_metrics(rdir)

        for suffix, name, value, split, slc, ro in suite_recs + ro_recs:
            mid = f"metric:{run}:{suffix}"
            if suffix.startswith("proto:"):
                art = ("master_benchmark_summary.json"
                       if os.path.exists(os.path.join(rdir, "master_benchmark_summary.json"))
                       else "evaluation_results/master_benchmark_summary.json")
            elif suffix.startswith("suite:") and ":sensor:" not in suffix:
                art = "evaluation_results/master_benchmark_summary.json"
            elif ":sensor:" in suffix and ro == "suite":
                art = "evaluation_results/master_benchmark_metrics.csv"
            else:
                art = ro_path or "readout_diagnostics.csv"
            aid = f"artifact:{run}:{os.path.basename(art)}"
            upsert_node(cur, aid, "Artifact", f"{run}/{os.path.basename(art)}",
                        {"kind": "eval"}, source=os.path.join(rdir, art), commit=commit)
            upsert_node(cur, mid, "Metric", f"{run} {split}/{slc} {name}={value:.4f}",
                        {"value": value, "metric": name, "split": split,
                         "slice": slc, "readout": ro}, source=os.path.join(rdir, art),
                        commit=commit)
            add_edge(cur, mid, aid, "MEASURED_ON", "parsed value", art, commit=commit)
            add_edge(cur, run, mid, "EVALUATED_WITH", "evaluation metric", art, commit=commit)
            n_metrics += 1

        for suffix, name, value, split, slc, ro, audit_path in audit_recs:
            mid = f"metric:{run}:{suffix}"
            aid = f"artifact:{run}:{os.path.basename(audit_path)}"
            upsert_node(cur, aid, "Artifact", f"{run}/{os.path.basename(audit_path)}",
                        {"kind": "audit"}, source=audit_path, commit=commit)
            upsert_node(cur, mid, "Metric", f"{run} {split}/{slc} {name}={value:.4f}",
                        {"value": value, "metric": name, "split": split,
                         "slice": slc, "readout": ro}, source=audit_path, commit=commit)
            add_edge(cur, mid, aid, "MEASURED_ON", "parsed audit metric", audit_path, commit=commit)
            add_edge(cur, run, mid, "EVALUATED_WITH", "audit metric", audit_path, commit=commit)
            n_metrics += 1

        wo_scan_recs = parse_wo_benchmark_metrics(rdir, run)
        for suffix, name, value, split, slc, ro, csv_p, extra_attrs in wo_scan_recs:
            mid = f"metric:{run}:{suffix}"
            aid = f"artifact:{run}:{os.path.basename(csv_p)}"
            upsert_node(cur, aid, "Artifact", f"{run}/{os.path.basename(csv_p)}",
                        {"kind": "per_scan_metrics"}, source=csv_p, commit=commit)
            m_attrs = {
                "value": value, "metric": name, "split": split, "slice": slc, "readout": ro,
                **extra_attrs
            }
            upsert_node(cur, mid, "Metric", f"{run} {extra_attrs.get('file','')} {name}={value:.4f}",
                        m_attrs, source=csv_p, commit=commit)
            add_edge(cur, mid, aid, "MEASURED_ON", "per-scan metric from csv", csv_p, commit=commit)
            add_edge(cur, run, mid, "EVALUATED_WITH", "per-scan metric", csv_p, commit=commit)
            n_metrics += 1

        if ro_ver:
            add_edge(cur, run, "R-P0", "EVALUATED_WITH", f"readout matrix {ro_ver}",
                     ro_path or "")

        t = next((v for sfx, n, v, sp, sl, ro in suite_recs
                  if sfx in ("suite:test:linear_probe_auc_mean", "proto:exp71a_concat:linear_probe:test:auc_mean",
                             "proto:jepa_pretrained_128:test:whitened_lr_auc_mean",
                             "proto:jepa_pretrained_128:test:raw_lr_auc_mean")), None)
        p1u = next((v for sfx, n, v, sp, sl, ro in ro_recs
                    if n == "auc_p1u_s2" and sp == "test" and sl == "all"), None)
        report.append((run, status, t, p1u, ro_ver))

    n_metrics += ingest_campaigns(cur)
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
    ap.add_argument("--extra-dirs", nargs="*", default=[
        os.path.join("src", "PECT_JEPA", "cross_condition", "experiments", "wo_20261028_registration"),
        os.path.join("src", "PECT_JEPA", "cross_condition", "experiments", "wo_20261027_r3"),
        os.path.join("src", "PECT_JEPA", "cross_condition", "experiments", "wo_20261027_r2"),
        os.path.join("src", "PECT_JEPA", "cross_condition", "experiments", "wo_20261027"),
    ])
    a = ap.parse_args()
    build(a.runs_dir, a.db, a.seed, extra_dirs=a.extra_dirs)
