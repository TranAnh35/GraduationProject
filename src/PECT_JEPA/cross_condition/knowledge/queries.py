"""Canned causal queries over the cross_condition knowledge graph.

Usage:
  C:\\Users\\tranl\\miniconda3\\envs\\ndt\\python.exe ^
    -m src.PECT_JEPA.cross_condition.knowledge.queries [Q1|Q2|Q3|ALL]

Q1: every run testing the invariance direction + verdict + cause.
Q2: best OOD readout under deployable (unsupervised-calibration) recipes.
Q3: dead slice: per-sensor test AUC across runs and key readouts.
"""

import json
import os
import sqlite3
import sys

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph.sqlite")


def q1(cur):
    print("== Q1: invariance-direction runs: verdict + cause ==")
    rows = cur.execute(
        "SELECT id, label, attrs FROM nodes WHERE type='Run' ORDER BY id").fetchall()
    for rid, label, attrs in rows:
        a = json.loads(attrs)
        cfg = a.get("config") or {}
        inv = cfg.get("inv_weight", "?") if cfg else "?"
        tested = cur.execute(
            "SELECT dst FROM edges WHERE src=? AND rel='TESTS'", (rid,)).fetchall()
        try:
            has_inv = float(inv) > 0
        except (TypeError, ValueError):
            has_inv = False
        if not tested and not has_inv:
            continue
        rej = cur.execute(
            "SELECT dst, rationale FROM edges WHERE src=? AND rel='REJECTED_BECAUSE'", (rid,)).fetchall()
        rej += cur.execute(
            "SELECT e.dst, e.rationale FROM edges e JOIN edges t ON t.dst=e.src "
            "WHERE t.src=? AND t.rel='TESTS' AND e.rel='REJECTED_BECAUSE'", (rid,)).fetchall()
        t = cur.execute(
            "SELECT attrs FROM nodes WHERE id=?", (f"metric:{rid}:suite:test:linear_probe_auc_mean",)).fetchone()
        tv = json.loads(t[0])["value"] if t else None
        print(f"- {rid} [{a.get('status')}] inv_w={inv} "
              f"pred={cfg.get('inv_predictor')} oracle={cfg.get('inv_oracle_mask')} "
              f"testLP={tv:.4f}" if tv is not None else f"- {rid} [{a.get('status')}] (no eval yet)")
        print(f"    verdict: {a.get('verdict', '')}")
        print(f"    tests: {', '.join(d[0] for d in tested) or '-'}")
        for dst, rat in rej:
            print(f"    REJECTED_BECAUSE {dst}: {rat[:160]}")


def q2(cur):
    print("== Q2: deployable readout ranking (test means, lineage reference first) ==")
    order = ["exp49_inv_xcov_20261014_021809", "exp51_asyminv_20261015_013636"]
    extra = [r[0] for r in cur.execute(
        "SELECT DISTINCT substr(id, 8, instr(substr(id, 8), ':')-1) FROM nodes "
        "WHERE id LIKE 'metric:%:readout:%'").fetchall()]
    for rid in order + [e for e in sorted(set(extra)) if e not in order]:
        rows = cur.execute(
            "SELECT id, label, attrs FROM nodes WHERE id LIKE ? AND type='Metric'",
            (f"metric:{rid}:readout:%:test:auc_%",)).fetchall()
        test_all = [r for r in rows if ":sensor:" not in r[0]]
        if not test_all:
            print(f"- {rid}: no readout matrix yet")
            continue
        print(f"- {rid}:")
        for mid, label, attrs in sorted(test_all, key=lambda r: -json.loads(r[2])["value"]):
            a = json.loads(attrs)
            ver = mid.split(":")[3] if mid.startswith("metric:") else "?"
            flag = {"auc_oracle": "(oracle-UB, needs GT)", "auc_p1_s2": "(needs GT bg)",
                    "auc_p1": "(needs GT bg, unshrunk)",
                    "auc_p1u": "<DEPLOYABLE, unshrunk bg (eps=1e-4)>",
                    "auc_p1u_s5": "<DEPLOYABLE, full-rank trimmed bg (eps=1e-5, keep_q=0.85)>",
                    "auc_p1u_maha": "<DEPLOYABLE, P1u + Mahalanobis z-score fusion>",
                    "auc_p1u_s2": "<DEPLOYABLE, shrunk s2>",
                    "auc_p1u_trim_s2": "<DEPLOYABLE, trimmed bg>",
                    "auc_mlp_p1u_s2": "(supervised probe)",
                    "auc_mlp_raw": "(supervised probe)"}.get(a['metric'], "")
            print(f"    {ver:>3s} {a['metric']:16s} {a['value']:.4f} {flag}")


def q3(cur):
    print("== Q3: dead slice check: test AUC per sensor (suite LP + key readouts) ==")
    runs = [r[0] for r in cur.execute(
        "SELECT id FROM nodes WHERE type='Run' ORDER BY id").fetchall()]
    header = f"{'run':32s} {'Hall_Air':>9s} {'Hall_Pot':>9s} {'TMR':>9s}  metric"
    print(header)
    for rid in runs:
        for pat, tag in (("suite:test:sensor:%:lp_auc", "suiteLP"),
                         ("readout:%:test:sensor:%:auc_p1u", "P1u"),
                         ("readout:%:test:sensor:%:auc_p1u_s5", "P1u_s5"),
                         ("readout:%:test:sensor:%:auc_p1u_maha", "P1u_maha"),
                         ("readout:%:test:sensor:%:auc_p1u_s2", "P1u_s2"),
                         ("readout:%:test:sensor:%:maha_auc", "maha")):
            rows = cur.execute(
                "SELECT id, attrs FROM nodes WHERE id LIKE ? AND type='Metric'",
                (f"metric:{rid}:{pat}",)).fetchall()
            if not rows:
                continue
            by = {}
            for mid, attrs in rows:
                a = json.loads(attrs)
                by[a["slice"]] = a["value"]
            if by:
                print(f"{rid[:32]:32s} {by.get('Hall_Air_Core', float('nan')):9.3f} "
                      f"{by.get('Hall_Pot_Core', float('nan')):9.3f} "
                      f"{by.get('TMR', float('nan')):9.3f}  {tag}")


def main():
    if not os.path.exists(DB):
        print(f"DB not found: {DB}. Run build.py first.")
        sys.exit(1)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    which = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    if which in ("Q1", "ALL"):
        q1(cur)
        print()
    if which in ("Q2", "ALL"):
        q2(cur)
        print()
    if which in ("Q3", "ALL"):
        q3(cur)
    con.close()


if __name__ == "__main__":
    main()
