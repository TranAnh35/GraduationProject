"""
Readout Diagnostics for Cross-Condition PECT-JEPA (no retraining).

Step 1 — Polarity-agnostic readout: the trained global linear probe fails on
unseen physics (Chirp/TMR score BELOW chance) while the sign-free Mahalanobis
distance stays above chance. This script measures, per scan:
  - auc_raw   : probe as-is
  - auc_oracle: max(auc, 1-auc) — upper bound if the sign were fixed
  - auc_agree : sign chosen UNSUPERVISED by agreement with the Mahalanobis
                anomaly map (Pearson corr of signed decision scores).
If auc_agree ~= auc_oracle >> auc_raw on OOD, the failure is a sign flip and
is fixable without retraining the encoder.

Step 2 — IoU headroom + probe strength sweep:
  - iou_05 / iou_opt (+ optimal threshold): separates "weak representation"
    from "bad fixed threshold".
  - auc_C025 / auc_C10 / auc_C40: LogisticRegression C sweep (same features).

Protocol mirrors evaluate.py STEP 1 (same file order, RNG seed, subsampling)
so the base probe is identical to the official benchmark.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import csv
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig
from src.PECT_JEPA.cross_condition.models.cross_condition_jepa import CrossConditionJEPA
from src.PECT_JEPA.cross_condition.data.preprocessing import (
    load_canonical_cscan_from_tdms,
    get_canonical_transformer,
)
from src.PECT_JEPA.cross_condition.data.split import (
    extract_file_metadata,
    get_canonical_train_val_test_splits,
)
from src.PECT_JEPA.cross_condition.evaluate import (
    resolve_paths,
    load_ground_truth_masks,
    build_corrosion_diameter_map,
    _iter_with_prefetch,
)
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Polarity + IoU readout diagnostics (no retraining)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--eval_splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--prefetch", type=lambda v: v.lower() == "true", default=True)
    parser.add_argument("--self_calibration", type=lambda v: v.lower() == "true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    ckpt_path, eval_out = resolve_paths(args)
    out_dir = args.out_dir or os.path.join(eval_out, "diagnose_readouts")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_config = ckpt.get("config", None)
    if isinstance(raw_config, CrossConditionConfig):
        config = raw_config
    else:
        config = CrossConditionConfig()
        if raw_config is not None:
            source_dict = vars(raw_config) if hasattr(raw_config, "__dict__") else (raw_config if isinstance(raw_config, dict) else {})
            for k, v in source_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
    model = CrossConditionJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    extractor = CScanFeatureExtractor(model=model, device=device, batch_size=args.batch_size)
    suite = DownstreamBenchmarkSuite(n_splits=5, random_state=42, device=device)
    gt_masks = load_ground_truth_masks()

    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]
    splits = get_canonical_train_val_test_splits(all_files, val_files_count=4, seed=42)
    get_canonical_transformer()

    # ---- STEP 1 (mirror evaluate.py): fit base readouts on train pool ----
    train_eval_files = [fp for fp in splits["train"]
                        if extract_file_metadata(fp)["specimen"] in gt_masks]
    cached: Dict[str, np.ndarray] = {}
    det_x, det_y = [], []
    rng = np.random.RandomState(42)
    for fp, cscan in tqdm(_iter_with_prefetch(train_eval_files, args.prefetch),
                          total=len(train_eval_files), desc="STEP1 pool", dynamic_ncols=True):
        sp = extract_file_metadata(fp)["specimen"]
        latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
        cached[fp] = latents
        flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
        flat_y = gt_masks[sp]["mask"].reshape(-1).astype(np.int64)
        pos_idx = np.where(flat_y == 1)[0]
        neg_idx = np.where(flat_y == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            sub_n = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
            sel = np.concatenate([pos_idx, rng.choice(neg_idx, size=sub_n, replace=False)])
            det_x.append(flat_z[sel])
            det_y.append(flat_y[sel])
    X_det = np.concatenate(det_x, axis=0)
    y_det = np.concatenate(det_y, axis=0)
    print(f"Pool: {len(y_det)} samples ({np.sum(y_det==1)} defect)", flush=True)

    base_readouts = suite.fit_global_readouts(X_det=X_det, y_det=y_det, X_size=None, y_size=None)
    s0, lr0 = base_readouts["scaler_det"], base_readouts["lr_probe"]
    # C-sweep reuses the SAME standardization so only C differs (fitting on
    # s0-transformed pool keeps the variants strictly comparable).
    Xs0 = s0.transform(X_det)
    lr025 = LogisticRegression(C=0.25, max_iter=500, tol=1e-3, class_weight="balanced",
                               random_state=42, solver="lbfgs").fit(Xs0, y_det)
    lr400 = LogisticRegression(C=4.0, max_iter=500, tol=1e-3, class_weight="balanced",
                               random_state=42, solver="lbfgs").fit(Xs0, y_det)

    def auc_of(lr, feats_s, y):
        p = lr.predict_proba(feats_s)[:, 1]
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0

    # ---- STEP 2: per-file diagnostics across splits ----
    rows: List[Dict[str, Any]] = []
    for split_name in args.eval_splits:
        eval_files = [fp for fp in splits.get(split_name, [])
                      if extract_file_metadata(fp)["specimen"] in gt_masks]
        uncached = [fp for fp in eval_files
                    if not (split_name == "train" and fp in cached)]
        stream = iter(_iter_with_prefetch(uncached, args.prefetch))
        nxt = next(stream, None)
        pbar = tqdm(eval_files, desc=f"Diagnosing [{split_name.upper()}]", dynamic_ncols=True)
        for fp in pbar:
            meta = extract_file_metadata(fp)
            gt_mask = gt_masks[meta["specimen"]]["mask"]
            if split_name == "train" and fp in cached:
                latents = cached[fp]
            else:
                assert nxt is not None and nxt[0] == fp
                latents = extractor.extract_full_cscan(nxt[1], self_calibrate=args.self_calibration)
                nxt = next(stream, None)

            flat = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
            y = gt_mask.reshape(-1).astype(np.int64)
            # GT masks may contain -1 (transition buffer): supervised metrics
            # only on valid pixels, mirroring evaluate_global_detection.
            valid = np.where(y >= 0)[0]
            yv = y[valid]
            feats_s = s0.transform(flat)
            feats_sv = feats_s[valid]
            dec = lr0.decision_function(feats_sv)
            auc_raw = auc_of(lr0, feats_sv, yv)
            ap_raw = float(average_precision_score(yv, lr0.predict_proba(feats_sv)[:, 1])) if len(np.unique(yv)) > 1 else 0.0
            auc_oracle = max(auc_raw, 1.0 - auc_raw)

            anom = suite.compute_anomaly_detection_metrics(latents, gt_mask)
            m = anom["anomaly_scores"].reshape(-1)[valid]
            c_pos = float(np.corrcoef(dec, m)[0, 1]) if np.std(dec) > 0 and np.std(m) > 0 else 0.0
            agree_sign = 1 if c_pos >= 0 else -1
            auc_agree = auc_raw if agree_sign == 1 else 1.0 - auc_raw
            # Rank-based variant: Spearman is outlier-robust where Pearson
            # flips on heavy-tailed anomaly maps.
            try:
                c_sp = float(spearmanr(dec, m).statistic)
            except Exception:
                c_sp = 0.0
            if not np.isfinite(c_sp):
                c_sp = 0.0
            agree_sp_sign = 1 if c_sp >= 0 else -1
            auc_agree_sp = auc_raw if agree_sp_sign == 1 else 1.0 - auc_raw

            prob_map = lr0.predict_proba(feats_s)[:, 1].reshape(gt_mask.shape)
            iou05 = suite.benchmark_segmentation_iou(prob_map, gt_mask, threshold=0.5)
            iouopt = suite.benchmark_segmentation_iou(prob_map, gt_mask, threshold=None)

            auc_c025 = auc_of(lr025, feats_sv, yv)
            auc_c400 = auc_of(lr400, feats_sv, yv)

            rows.append({
                "file": os.path.basename(fp), "split": split_name,
                "specimen": meta["specimen"], "sensor": meta["sensor"],
                "waveform": meta["waveform"], "liftoff": meta["liftoff"],
                "auc_raw": round(auc_raw, 4), "ap_raw": round(ap_raw, 4),
                "auc_oracle": round(auc_oracle, 4), "auc_agree": round(auc_agree, 4),
                "auc_agree_sp": round(auc_agree_sp, 4),
                "agree_sign": agree_sign, "agree_sp_sign": agree_sp_sign,
                "iou_05": iou05["iou"], "iou_opt": iouopt["iou"],
                "opt_thresh": iouopt["optimal_threshold"],
                "auc_C025": round(auc_c025, 4), "auc_C10": round(auc_raw, 4),
                "auc_C40": round(auc_c400, 4),
                "maha_auc": anom["auc_roc"], "cnr": anom["cnr"],
            })
            pbar.set_postfix({"raw": f"{auc_raw:.3f}", "agr": f"{auc_agree:.3f}",
                              "iou05": f"{iou05['iou']:.3f}", "iouOpt": f"{iouopt['iou']:.3f}"})

    csv_path = os.path.join(out_dir, "readout_diagnostics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def mean(key, rs):
        v = [r[key] for r in rs if isinstance(r[key], (int, float))]
        return float(np.mean(v)) if v else 0.0

    print("\n==== READOUT DIAGNOSTICS ====", flush=True)
    for sp in args.eval_splits:
        rs = [r for r in rows if r["split"] == sp]
        print(f"[{sp.upper():5s}] n={len(rs)} raw={mean('auc_raw',rs):.4f} oracle={mean('auc_oracle',rs):.4f} "
              f"agree={mean('auc_agree',rs):.4f} agreeSP={mean('auc_agree_sp',rs):.4f} "
              f"iou05={mean('iou_05',rs):.4f} iouOpt={mean('iou_opt',rs):.4f} "
              f"C025={mean('auc_C025',rs):.4f} C40={mean('auc_C40',rs):.4f} maha={mean('maha_auc',rs):.4f}", flush=True)
    for grp in ["sensor", "waveform"]:
        print(f"--- by {grp} (raw / agreeSP / oracle):", flush=True)
        keys = sorted(set(r[grp] for r in rows))
        for k in keys:
            rs = [r for r in rows if r[grp] == k]
            print(f"  {k:15s} n={len(rs):2d} {mean('auc_raw',rs):.3f} / {mean('auc_agree_sp',rs):.3f} / {mean('auc_oracle',rs):.3f}", flush=True)
    print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
