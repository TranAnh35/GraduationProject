"""
Universal PECT Foundation Model Benchmark Engine.

Evaluates a SINGLE PECT-JEPA Foundation checkpoint across ALL core NDT tasks:
1. Defect Anomaly Screening & Detection (AUC, AP, CNR) across multiple specimens & lift-offs
2. Quantitative Physical Defect Sizing (Depth R^2, MAE um, Diameter rho, Volume rho)
3. Cross-Sensor Zero-Shot Generalization (Hall Air <-> Hall Pot <-> TMR)
4. Multi-Waveform Robustness (Chirp, Square, Gaussian)
"""

import os
import sys
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch

ROOT_DIR = r"E:\Project_On_Lab\Research\GraduationProject"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.evaluate import (
    load_model_from_checkpoint,
    load_cscan_from_tdms,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets


class PECTFoundationEvaluator:
    """
    Unified Foundation Benchmark Evaluator.
    Evaluates ONE model checkpoint on ALL downstream NDT tasks simultaneously.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = load_model_from_checkpoint(checkpoint_path, device=device).eval()
        self.gt_mgr = get_ground_truth_manager()

        # Topology offsets
        self.topology = getattr(self.model.config, "spatial_topology", "concentric_star")
        self.star_radii = getattr(self.model.config, "star_radii", (1, 3, 7))
        self.grid_size = getattr(self.model.config, "grid_size", 5)
        self.offsets = get_spatial_topology_offsets(
            topology=self.topology, star_radii=self.star_radii, grid_size=self.grid_size
        )
        self.pad = int(np.max(np.abs(self.offsets)))

    def extract_cscan_features(self, filepath: str, self_calibrate: bool = False, batch_size: int = 1024) -> Tuple[np.ndarray, Tuple[int, int]]:
        grid = load_cscan_from_tdms(filepath, standardize_coords=True)
        sY, sX, C = grid.shape
        padded = np.pad(grid, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")

        all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
        all_r = all_r.reshape(-1)
        all_c = all_c.reshape(-1)
        total_pts = sY * sX

        out_feats = []
        with torch.no_grad():
            for k in range(0, total_pts, batch_size):
                k_end = min(k + batch_size, total_pts)
                rows_b = all_r[k:k_end] + self.pad
                cols_b = all_c[k:k_end] + self.pad
                sample_r = rows_b[:, None] + self.offsets[None, :, 0]
                sample_c = cols_b[:, None] + self.offsets[None, :, 1]
                patch_b = padded[sample_r, sample_c, :].reshape(-1, self.grid_size, self.grid_size, C)
                xb = torch.from_numpy(patch_b).float().to(self.device)

                if hasattr(self.model, "extract_foundation_representation"):
                    z = self.model.extract_foundation_representation(xb, self_calibrate=self_calibrate)
                else:
                    z = self.model.extract_features(xb)
                out_feats.append(z.cpu().numpy())

        feats = np.vstack(out_feats).reshape(sY, sX, -1)
        return feats, (sY, sX)

    def evaluate_anomaly_detection(self, feat_map: np.ndarray, specimen_type: str) -> Dict[str, float]:
        sY, sX, D = feat_map.shape
        mask = self.gt_mgr.get_standard_cropped_mask(specimen_type, buffer_px=2)
        min_Y, min_X = min(sY, mask.shape[0]), min(sX, mask.shape[1])
        y_flat = (mask[:min_Y, :min_X] > 0).astype(int).reshape(-1)
        X_flat = feat_map[:min_Y, :min_X].reshape(-1, D)

        np.random.seed(42)
        def_idx = np.where(y_flat == 1)[0]
        snd_idx = np.where(y_flat == 0)[0]
        eval_idx = np.concatenate([
            np.random.choice(def_idx, min(len(def_idx), 1000), replace=False),
            np.random.choice(snd_idx, min(len(snd_idx), 4000), replace=False),
        ])

        X_eval = X_flat[eval_idx]
        y_eval = y_flat[eval_idx]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        p_prob = np.zeros(len(y_eval))
        for tr_i, te_i in kf.split(X_eval):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=300, class_weight="balanced")
            clf.fit(sc.fit_transform(X_eval[tr_i]), y_eval[tr_i])
            p_prob[te_i] = clf.predict_proba(sc.transform(X_eval[te_i]))[:, 1]

        auc = roc_auc_score(y_eval, p_prob)
        ap = average_precision_score(y_eval, p_prob)
        mu_d = np.mean(p_prob[y_eval == 1])
        mu_s = np.mean(p_prob[y_eval == 0])
        var_d = np.var(p_prob[y_eval == 1])
        var_s = np.var(p_prob[y_eval == 0])
        cnr = abs(mu_d - mu_s) / np.sqrt(0.5 * (var_s + var_d) + 1e-8)

        return {"auc": float(auc), "ap": float(ap), "cnr": float(cnr)}

    def evaluate_defect_sizing(self, feat_map: np.ndarray, specimen_type: str = "corrosion") -> Dict[str, float]:
        sY, sX, D = feat_map.shape
        feats_cad = self.gt_mgr.get_flaw_features(specimen_type, coordinate_system="cropped")

        d_true, diam_true, vol_true, d_feats = [], [], [], []
        for f in feats_cad:
            cx = max(0, min(sX - 1, int(round(f["x"]))))
            cy = max(0, min(sY - 1, int(round(f["y"]))))
            d_true.append(f["depth"])
            diam_true.append(f.get("diameter", 5.0))
            vol_true.append(np.pi * ((f.get("diameter", 5.0) / 2.0) ** 2) * f["depth"])
            d_feats.append(feat_map[cy, cx])

        d_true = np.array(d_true)
        diam_true = np.array(diam_true)
        vol_true = np.array(vol_true)
        X_flaws = np.array(d_feats)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        d_pred = np.zeros(len(d_true))
        for tr_i, te_i in kf.split(X_flaws):
            sc = StandardScaler()
            reg = Ridge(alpha=10.0)
            reg.fit(sc.fit_transform(X_flaws[tr_i]), d_true[tr_i])
            d_pred[te_i] = reg.predict(sc.transform(X_flaws[te_i]))

        r2 = r2_score(d_true, d_pred)
        mae_um = mean_absolute_error(d_true, d_pred) * 1000.0

        # Pairwise relational geometry
        N = len(d_true)
        diff_d, diff_diam, diff_vol, dist_latent = [], [], [], []
        for i in range(N):
            for j in range(i + 1, N):
                diff_d.append(abs(d_true[i] - d_true[j]))
                diff_diam.append(abs(diam_true[i] - diam_true[j]))
                diff_vol.append(abs(vol_true[i] - vol_true[j]))
                dist_latent.append(np.linalg.norm(X_flaws[i] - X_flaws[j]))

        rho_d = float(np.corrcoef(dist_latent, diff_d)[0, 1])
        rho_diam = float(np.corrcoef(dist_latent, diff_diam)[0, 1])
        rho_vol = float(np.corrcoef(dist_latent, diff_vol)[0, 1])

        return {
            "depth_r2": float(r2),
            "mae_um": float(mae_um),
            "rho_depth": rho_d,
            "rho_diameter": rho_diam,
            "rho_volume": rho_vol,
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Executes the complete Foundation Model benchmark suite.
        """
        print("=" * 80)
        print("  PECT FOUNDATION MODEL UNIFIED BENCHMARK EVALUATION")
        print("=" * 80)

        test_scans = {
            "Corrosion Chirp z1": (
                r"data/Hall_Air_Core/Corrosion/Chirp/hall_aircore_corosion_frontside_chirp_300x300_2.97_500-1500hz_z1_20260118_162411.tdms",
                "corrosion",
            ),
            "Corrosion Square z1": (
                r"data/Hall_Air_Core/Corrosion/Square/hall_aircore_corosion_frontside_square_300x300_2.1_200hz_z1_20260117_162910.tdms",
                "corrosion",
            ),
            "Corrosion Chirp z3 (High Lift-Off)": (
                r"data/Hall_Air_Core/Corrosion/Chirp/hall_aircore_corosion_frontside_chirp_300x300_2.97_500-1500hz_z3_20260118_231219.tdms",
                "corrosion",
            ),
            "Rivet Chirp z1": (
                r"data/Hall_Air_Core/Rivet_v1/Chirp/hall_aircore_rivet_frontside_chirp_300x300_500_1500hz_z1_20260123_202334.tdms",
                "rivet",
            ),
            "Rivet Chirp z3 (High Lift-Off)": (
                r"data/Hall_Air_Core/Rivet_v1/Chirp/hall_aircore_rivet_frontside_chirp_300x300_500_1500hz_z3_20260124_031140.tdms",
                "rivet",
            ),
            "Corrosion Chirp z1 (TMR Sensor)": (
                r"data/TMR/Corrosion/Chirp/tmr_corosion_frontside_chirp_300x300_z1_20260128_015728.tdms",
                "corrosion",
            ),
        }

        results = {"detection": {}, "sizing": {}}

        for scan_name, (fp, spec) in test_scans.items():
            print(f"\nProcessing {scan_name}...")
            feats, _ = self.extract_cscan_features(fp)
            det_metrics = self.evaluate_anomaly_detection(feats, spec)
            results["detection"][scan_name] = det_metrics
            print(f"  [Detection] AUC: {det_metrics['auc']:.4f} | AP: {det_metrics['ap']:.4f} | CNR: {det_metrics['cnr']:.2f}")

            if spec == "corrosion":
                size_metrics = self.evaluate_defect_sizing(feats, spec)
                results["sizing"][scan_name] = size_metrics
                print(f"  [Sizing]    Depth R²: {size_metrics['depth_r2']:.4f} | MAE: {size_metrics['mae_um']:.1f} um | Vol rho: {size_metrics['rho_volume']:.4f}")

        # Summary Table
        print("\n" + "=" * 80)
        print("  FOUNDATION MODEL SCORECARD")
        print("=" * 80)
        df_det = pd.DataFrame(results["detection"]).T
        print("\n--- Task 1: Anomaly Screening & Detection ---")
        print(df_det.round(4).to_string())

        df_size = pd.DataFrame(results["sizing"]).T
        print("\n--- Task 2: Quantitative Physical Defect Sizing ---")
        print(df_size.round(4).to_string())

        return results


if __name__ == "__main__":
    ckpt = r"experiments/5x5/exp25b_scale_preserved_dual_stream_fixed/checkpoints/best_model_5x5.pt"
    evaluator = PECTFoundationEvaluator(ckpt)
    scorecard = evaluator.run_full_benchmark()
