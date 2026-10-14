"""
Readout Diagnostics v2 for Cross-Condition PECT-JEPA (no retraining).

Visual evidence (train probe map) shows BROAD structured background false
positives while the scan-adaptive Mahalanobis map stays clean: a fixed global
probe cannot suppress scan-specific background structures. This script tests
two no-retraining fixes on the SAME frozen latents:

  P0 base      : raw features, random sound negatives (official recipe)
  P1 whitened  : per-scan background-whitened features, random negatives.
                 Bg stats (mu, inv-sqrt-cov) from pseudo-sound pixels
                 (valid labels + Mahalanobis < 80th pct, unsupervised cut).
                 Whitening suppresses scan-specific background covariance
                 (liftoff gradients, raster texture) before the probe.
  P2 hard-neg  : raw features, defects + top-2000/scan P0 false alarms
                 (structured FP regions) + random fill.
  P3 whitened + hard-neg.

Plus (unchanged from v1): polarity oracle/agreement, IoU@0.5 vs optimal,
probe C-sweep. STEP 1 mirrors evaluate.py exactly (order, seed, subsample)
so P0 is identical to the official benchmark.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import csv
import argparse
from typing import Dict, List, Any
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
    _iter_with_prefetch,
)
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

BG_Q = 0.80       # pseudo-sound cut: Mahalanobis below 80th pct
HARD_K = 2000     # hard negatives per train scan


def parse_args():
    parser = argparse.ArgumentParser(description="Scan-adaptive + hard-neg readout diagnostics (no retraining)")
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


def fit_lr(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 42):
    lr = LogisticRegression(C=C, max_iter=500, tol=1e-3, class_weight="balanced",
                            random_state=seed, solver="lbfgs")
    lr.fit(X, y)
    return lr


def whiten_stats(bg: np.ndarray) -> tuple:
    """(mu, W) with W = inv-sqrt(cov + eps*I) via eigh. bg: [Nb, D]."""
    mu = bg.mean(axis=0)
    cov = np.cov(bg, rowvar=False) + np.eye(bg.shape[1], dtype=np.float64) * 1e-4
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    W = (vecs / np.sqrt(vals)).dot(vecs.T).astype(np.float32)
    return mu.astype(np.float32), W


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

    # ---- STEP 1 (mirror evaluate.py): pool + P0 base probe ----
    train_eval_files = [fp for fp in splits["train"]
                        if extract_file_metadata(fp)["specimen"] in gt_masks]
    cached: Dict[str, np.ndarray] = {}
    gt_of: Dict[str, np.ndarray] = {}
    pool_parts: List[Dict[str, Any]] = []  # per-scan (fp, X_sel, y_sel) w/ scan id
    det_x, det_y = [], []
    rng = np.random.RandomState(42)
    for fp, cscan in tqdm(_iter_with_prefetch(train_eval_files, args.prefetch),
                          total=len(train_eval_files), desc="STEP1 pool", dynamic_ncols=True):
        sp = extract_file_metadata(fp)["specimen"]
        gt = gt_masks[sp]["mask"]
        latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
        cached[fp] = latents
        gt_of[fp] = gt
        flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
        flat_y = gt.reshape(-1).astype(np.int64)
        pos_idx = np.where(flat_y == 1)[0]
        neg_idx = np.where(flat_y == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            sub_n = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
            sel = np.concatenate([pos_idx, rng.choice(neg_idx, size=sub_n, replace=False)])
            det_x.append(flat_z[sel])
            det_y.append(flat_y[sel])
            pool_parts.append({"fp": fp, "X": flat_z[sel], "y": flat_y[sel]})
    X_det = np.concatenate(det_x, axis=0)
    y_det = np.concatenate(det_y, axis=0)
    print(f"Pool: {len(y_det)} samples ({np.sum(y_det==1)} defect)", flush=True)

    base_readouts = suite.fit_global_readouts(X_det=X_det, y_det=y_det, X_size=None, y_size=None)
    s0, lr0 = base_readouts["scaler_det"], base_readouts["lr_probe"]
    Xs0 = s0.transform(X_det)
    lr025 = fit_lr(Xs0, y_det, C=0.25)
    lr400 = fit_lr(Xs0, y_det, C=4.0)

    # ---- STEP 1b: per-train-scan bg stats + P1/P2/P3 probes ----
    # Pseudo-sound = valid labels + Mahalanobis below BG_Q (defects are only
    # ~1-3%, far below the 20% cut, so bg stays nearly pure).
    bg_stats: Dict[str, tuple] = {}
    for fp in tqdm(train_eval_files, desc="bg stats", dynamic_ncols=True):
        anom = suite.compute_anomaly_detection_metrics(cached[fp], gt_of[fp])
        m = anom["anomaly_scores"].reshape(-1)
        y = gt_of[fp].reshape(-1).astype(np.int64)
        valid = y >= 0
        bg_mask = valid & (m < np.quantile(m[valid], BG_Q))
        bg_stats[fp] = whiten_stats(cached[fp].reshape(-1, cached[fp].shape[-1])[bg_mask])

    def whiten_fp(fp: str, X: np.ndarray) -> np.ndarray:
        mu, W = bg_stats[fp]
        return ((X - mu) @ W).astype(np.float32)

    # P1: whitened pool, random negatives (same samples as P0)
    Xw_det = np.concatenate([whiten_fp(p["fp"], p["X"]) for p in pool_parts], axis=0)
    lr1 = fit_lr(Xw_det, y_det)

    # P2/P3: hard negatives = top-HARD_K P0 false alarms per scan + random fill
    hard_x, hard_fp = [], []
    for fp in train_eval_files:
        flat = cached[fp].reshape(-1, cached[fp].shape[-1]).astype(np.float32)
        y = gt_of[fp].reshape(-1).astype(np.int64)
        snd = np.where(y == 0)[0]
        if len(snd) == 0:
            continue
        p = lr0.predict_proba(s0.transform(flat[snd]))[:, 1]
        k = min(HARD_K, len(snd))
        topk = snd[np.argpartition(-p, k - 1)[:k]]
        hard_x.append(flat[topk])
        hard_fp += [fp] * k
    X_hard = np.concatenate(hard_x, axis=0)
    y_hard = np.zeros(len(X_hard), dtype=np.int64)
    pos_all = np.concatenate([p["X"][p["y"] == 1] for p in pool_parts], axis=0)
    pos_fp = []
    for p in pool_parts:
        pos_fp += [p["fp"]] * int(np.sum(p["y"] == 1))
    budget = max(len(pos_all) * 4, 2500)
    fill_n = max(0, min(budget - len(X_hard), 10 ** 9))
    # random fill from sound pixels not already taken as hard
    fill_x, fill_fp = [], []
    for fp in train_eval_files:
        if fill_n <= 0:
            break
        flat = cached[fp].reshape(-1, cached[fp].shape[-1]).astype(np.float32)
        y = gt_of[fp].reshape(-1).astype(np.int64)
        snd = np.where(y == 0)[0]
        take = min(len(snd), fill_n)
        if take <= 0:
            continue
        sel = rng.choice(snd, size=take, replace=False)
        fill_x.append(flat[sel])
        fill_fp += [fp] * take
        fill_n -= take
    X_neg = np.concatenate([X_hard] + fill_x, axis=0)
    neg_fp = hard_fp + fill_fp
    X_p2 = np.concatenate([pos_all, X_neg], axis=0)
    y_p2 = np.concatenate([np.ones(len(pos_all), dtype=np.int64),
                           np.zeros(len(X_neg), dtype=np.int64)])
    print(f"P2 pool: {len(y_p2)} ({len(pos_all)} pos, {len(X_hard)} hardneg, {len(X_neg)-len(X_hard)} randneg)", flush=True)
    lr2 = fit_lr(s0.transform(X_p2), y_p2)
    Xw_p2 = np.concatenate(
        [whiten_fp(fp, X_p2[i:i + 1]) for i, fp in enumerate(pos_fp + neg_fp)], axis=0)
    lr3 = fit_lr(Xw_p2, y_p2)

    def auc_of(lr, Xs, y):
        p = lr.predict_proba(Xs)[:, 1]
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0

    # ---- STEP 2: per-file diagnostics, all probes ----
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
            anom = None
            if split_name == "train" and fp in cached:
                latents, gt_mask = cached[fp], gt_of[fp]
                muW = bg_stats[fp]
            else:
                assert nxt is not None and nxt[0] == fp
                gt_mask = gt_masks[meta["specimen"]]["mask"]
                latents = extractor.extract_full_cscan(nxt[1], self_calibrate=args.self_calibration)
                nxt = next(stream, None)
                anom = suite.compute_anomaly_detection_metrics(latents, gt_mask)
                m0 = anom["anomaly_scores"].reshape(-1)
                y0 = gt_mask.reshape(-1).astype(np.int64)
                v0 = y0 >= 0
                bgm = v0 & (m0 < np.quantile(m0[v0], BG_Q))
                muW = whiten_stats(latents.reshape(-1, latents.shape[-1])[bgm])

            flat = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
            y = gt_mask.reshape(-1).astype(np.int64)
            valid = np.where(y >= 0)[0]
            yv = y[valid]
            feats_s = s0.transform(flat)
            feats_sv = feats_s[valid]
            flat_w = ((flat - muW[0]) @ muW[1]).astype(np.float32)
            flat_wv = flat_w[valid]

            dec = lr0.decision_function(feats_sv)
            auc_raw = auc_of(lr0, feats_sv, yv)
            ap_raw = float(average_precision_score(yv, lr0.predict_proba(feats_sv)[:, 1])) if len(np.unique(yv)) > 1 else 0.0
            auc_oracle = max(auc_raw, 1.0 - auc_raw)

            if anom is None:
                anom = suite.compute_anomaly_detection_metrics(latents, gt_mask)
            m = anom["anomaly_scores"].reshape(-1)[valid]
            c_pos = float(np.corrcoef(dec, m)[0, 1]) if np.std(dec) > 0 and np.std(m) > 0 else 0.0
            auc_agree = auc_raw if c_pos >= 0 else 1.0 - auc_raw
            try:
                c_sp = float(spearmanr(dec, m).statistic)
            except Exception:
                c_sp = 0.0
            if not np.isfinite(c_sp):
                c_sp = 0.0
            auc_agree_sp = auc_raw if c_sp >= 0 else 1.0 - auc_raw

            auc_p1 = auc_of(lr1, flat_wv, yv)
            ap_p1 = float(average_precision_score(yv, lr1.predict_proba(flat_wv)[:, 1])) if len(np.unique(yv)) > 1 else 0.0
            auc_p2 = auc_of(lr2, feats_sv, yv)
            ap_p2 = float(average_precision_score(yv, lr2.predict_proba(feats_sv)[:, 1])) if len(np.unique(yv)) > 1 else 0.0
            auc_p3 = auc_of(lr3, flat_wv, yv)
            ap_p3 = float(average_precision_score(yv, lr3.predict_proba(flat_wv)[:, 1])) if len(np.unique(yv)) > 1 else 0.0

            prob_map = lr0.predict_proba(feats_s)[:, 1].reshape(gt_mask.shape)
            iou05 = suite.benchmark_segmentation_iou(prob_map, gt_mask, threshold=0.5)
            iouopt = suite.benchmark_segmentation_iou(prob_map, gt_mask, threshold=None)
            prob3 = lr3.predict_proba(flat_w)[:, 1].reshape(gt_mask.shape)
            iou05_p3 = suite.benchmark_segmentation_iou(prob3, gt_mask, threshold=0.5)
            iouopt_p3 = suite.benchmark_segmentation_iou(prob3, gt_mask, threshold=None)

            auc_c025 = auc_of(lr025, feats_sv, yv)
            auc_c400 = auc_of(lr400, feats_sv, yv)

            rows.append({
                "file": os.path.basename(fp), "split": split_name,
                "specimen": meta["specimen"], "sensor": meta["sensor"],
                "waveform": meta["waveform"], "liftoff": meta["liftoff"],
                "auc_raw": round(auc_raw, 4), "ap_raw": round(ap_raw, 4),
                "auc_oracle": round(auc_oracle, 4), "auc_agree": round(auc_agree, 4),
                "auc_agree_sp": round(auc_agree_sp, 4),
                "auc_p1": round(auc_p1, 4), "ap_p1": round(ap_p1, 4),
                "auc_p2": round(auc_p2, 4), "ap_p2": round(ap_p2, 4),
                "auc_p3": round(auc_p3, 4), "ap_p3": round(ap_p3, 4),
                "iou_05": iou05["iou"], "iou_opt": iouopt["iou"],
                "opt_thresh": iouopt["optimal_threshold"],
                "iou05_p3": iou05_p3["iou"], "iouOpt_p3": iouopt_p3["iou"],
                "auc_C025": round(auc_c025, 4), "auc_C40": round(auc_c400, 4),
                "maha_auc": anom["auc_roc"], "cnr": anom["cnr"],
            })
            pbar.set_postfix({"p0": f"{auc_raw:.3f}", "p1": f"{auc_p1:.3f}",
                              "p2": f"{auc_p2:.3f}", "p3": f"{auc_p3:.3f}"})

    csv_path = os.path.join(out_dir, "readout_diagnostics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def mean(key, rs):
        v = [r[key] for r in rs if isinstance(r[key], (int, float))]
        return float(np.mean(v)) if v else 0.0

    print("\n==== READOUT DIAGNOSTICS v2 ====", flush=True)
    for sp in args.eval_splits:
        rs = [r for r in rows if r["split"] == sp]
        print(f"[{sp.upper():5s}] n={len(rs)} "
              f"P0={mean('auc_raw',rs):.4f}/{mean('ap_raw',rs):.4f} "
              f"P1={mean('auc_p1',rs):.4f}/{mean('ap_p1',rs):.4f} "
              f"P2={mean('auc_p2',rs):.4f}/{mean('ap_p2',rs):.4f} "
              f"P3={mean('auc_p3',rs):.4f}/{mean('ap_p3',rs):.4f} "
              f"maha={mean('maha_auc',rs):.4f} "
              f"iou05={mean('iou_05',rs):.4f}/{mean('iou05_p3',rs):.4f} "
              f"iouOpt={mean('iou_opt',rs):.4f}/{mean('iouOpt_p3',rs):.4f}", flush=True)
    for grp in ["sensor", "waveform"]:
        print(f"--- by {grp} AUC/AP (P0 | P1 | P2 | P3):", flush=True)
        for k in sorted(set(r[grp] for r in rows)):
            rs = [r for r in rows if r[grp] == k]
            print(f"  {k:15s} n={len(rs):2d} {mean('auc_raw',rs):.3f}/{mean('ap_raw',rs):.3f} | "
                  f"{mean('auc_p1',rs):.3f}/{mean('ap_p1',rs):.3f} | "
                  f"{mean('auc_p2',rs):.3f}/{mean('ap_p2',rs):.3f} | "
                  f"{mean('auc_p3',rs):.3f}/{mean('ap_p3',rs):.3f}", flush=True)
    print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
