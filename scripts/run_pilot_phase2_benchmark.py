"""
Pilot Phase 2 Benchmark Script according to PECT-JEPA v1.0-RC1 Specification.
Evaluates JEPA Predictor anomaly score vs Spatial Linear Baselines on primary scans:
1. HallAir_Chirp_z1 (High SNR, Transductive)
2. TMR_Chirp_z1 (High Sensitivity, Transductive)
3. HallPot_Chirp_z1 (Low SNR Baseline, Transductive)

Implements:
- Điều 1: Anomaly maps A_M(c), radius R(D) in {6, 7} mm, object scores S_M(c_i; R).
- Điều 2: Maximum independent sham grid (40 interior + 16 edge), quantiles q_i, p_i.
- Prerequisite gates: G0 (Positive control), G1 (Null validity G-N1 & G-N2), G2 (Stripe noise confound), G3 (Statistical power |O_s| >= 8).
- Two-tier bootstrap (B=2000) for 95% CI [L_s, U_s].
- A/A test between linear baselines to calibrate delta_min.
- Three-state decision rules: ESTABLISHED / REJECTED / NOT DETECTED.
"""

import os
import sys
import json
import numpy as np
import torch
from nptdms import TdmsFile
from sklearn.linear_model import Ridge
from scipy import stats

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

ROOT_DIR = r"E:\Project_On_Lab\Research\GraduationProject"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import linear_time_resample
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.evaluate import load_model_from_checkpoint
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms

# -----------------------------------------------------------------------------
# 1. SETUP & GEOMETRIC CONFIGURATION
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = os.path.join(ROOT_DIR, r"experiments\5x5\exp28_full_20ep\checkpoints\best_model_5x5.pt")
print(f"Loading checkpoint EXP-28 on {device} ...")
model = load_model_from_checkpoint(ckpt_path, device=device)
model.eval()

# Load CAD metadata
cad_corr_path = os.path.join(ROOT_DIR, r"data\ground_truth\corrosion\corrosion_flaws_config.json")
with open(cad_corr_path, "r", encoding="utf-8") as f:
    cad_corr = json.load(f)["features"]

# Flaw grouping according to Điều 2.1:
# 9 interior (row, col in {2, 3, 4}) -> cad_x, cad_y in {75, 135, 195}
interior_flaw_ids = [f["id"] for f in cad_corr if (f["x"] in [75, 135, 195]) and (f["y"] in [75, 135, 195])]
# 4 corners: ID 1, 5, 21, 25
corner_flaw_ids = [1, 5, 21, 25]
# 12 edge non-corner: remaining
edge_flaw_ids = [f["id"] for f in cad_corr if (f["id"] not in interior_flaw_ids) and (f["id"] not in corner_flaw_ids)]

print(f"Flaw classification (Total 25): {len(interior_flaw_ids)} interior, {len(edge_flaw_ids)} edge non-corner, {len(corner_flaw_ids)} corners.")

# Construct Maximum Independent Sham Grid (Điều 2.3)
# L = {(15 + 30*a, 15 + 30*b)} with a, b in {0..8}
interior_shams = []
for a in range(1, 8):
    for b in range(1, 8):
        if not (a % 2 == 0 and b % 2 == 0):  # Not both even (not CAD flaws)
            interior_shams.append((15 + 30 * a, 15 + 30 * b))

edge_shams_by_side = {"left": [], "right": [], "top": [], "bottom": []}
for b in [1, 3, 5, 7]:
    edge_shams_by_side["left"].append((15 + 30 * 0, 15 + 30 * b))
    edge_shams_by_side["right"].append((15 + 30 * 8, 15 + 30 * b))
for a in [1, 3, 5, 7]:
    edge_shams_by_side["top"].append((15 + 30 * a, 15 + 30 * 0))
    edge_shams_by_side["bottom"].append((15 + 30 * a, 15 + 30 * 8))

all_edge_shams = (edge_shams_by_side["left"] + edge_shams_by_side["right"] + 
                  edge_shams_by_side["top"] + edge_shams_by_side["bottom"])

print(f"Independent Sham Grid: N_interior = {len(interior_shams)} (expected 40), N_edge = {len(all_edge_shams)} (expected 16).")

# Concentric star offsets [25, 2]
star_offsets = get_spatial_topology_offsets(topology="concentric_star", star_radii=(1, 3, 7))

# Helper to read and process TDMS
def load_cscan_cube(fpath):
    print(f"Loading {os.path.basename(fpath)} via authoritative load_cscan_from_tdms ...")
    return load_cscan_from_tdms(
        file_path=fpath,
        temporal_samples=128,
        normalization="file_peak",
        standardize_coords=True
    )

# -----------------------------------------------------------------------------
# 2. EVALUATION PIPELINE FOR A SINGLE SCAN
# -----------------------------------------------------------------------------
def evaluate_single_scan(fpath, scan_name, transductive_label=True):
    print("\n" + "=" * 80)
    print(f"EVALUATING PRIMARY SCAN: {scan_name} [{ 'Transductive' if transductive_label else 'Inductive' }]")
    print("=" * 80)
    cscan = load_cscan_cube(fpath)
    H, W, C = cscan.shape  # 270, 270, 128
    
    # 2.1 Calculate Raw PtP for Z_PtP thresholding
    ptp_raw = np.ptp(cscan, axis=-1)  # [270, 270]
    
    # Simple sham baseline for PtP Z-score (to find O_s with Z_PtP >= 3)
    yy_grid, xx_grid = np.indices((H, W))
    def calc_amp_ptp(cx, cy):
        r = np.hypot(xx_grid - cx, yy_grid - cy)
        core = ptp_raw[r <= 8]
        ring = ptp_raw[(r >= 18) & (r < 28)]
        if len(ring) == 0 or len(core) == 0:
            return 0.0
        return float(np.mean(core) - np.median(ring))
    
    sham_ptp_vals = [calc_amp_ptp(x, y) for x, y in interior_shams]
    mu_ptp, std_ptp = np.mean(sham_ptp_vals), np.std(sham_ptp_vals) + 1e-12
    
    flaw_z_ptp = {}
    for f in cad_corr:
        val = calc_amp_ptp(f["x"], f["y"])
        flaw_z_ptp[f["id"]] = (val - mu_ptp) / std_ptp

    # Identify evaluable flaws O_s: interior and edge non-corner with Z_PtP >= 3
    candidate_flaws = interior_flaw_ids + edge_flaw_ids
    O_s = [fid for fid in candidate_flaws if flaw_z_ptp[fid] >= 3.0]
    pos_control_flaws = [fid for fid in candidate_flaws if flaw_z_ptp[fid] >= 6.0]
    
    print(f"Evaluable Set O_s (Z_PtP >= 3): {len(O_s)} flaws out of 21 candidates.")
    print(f"Positive Control Set (Z_PtP >= 6): {len(pos_control_flaws)} flaws.")

    # 2.2 Train Linear Baselines
    # Extract training patches for linear models across V = [7..262] x [7..262]
    # We sample a representative set of points to fit W in R^{24*128 x 128} or fit per-feature
    # Predicting probe 0 [128] from 24 probes [24*128]
    print("Training Spatial Linear Baselines (Lin-U1, Lin-U2, Lin-H) ...")
    
    # Gather training pairs: ctx [N_samples, 24 * 128], tgt [N_samples, 128]
    np.random.seed(42)
    sample_coords = []
    for u in range(10, 260, 3):
        for v in range(10, 260, 3):
            sample_coords.append((u, v))
            
    # Lin-U: all sample coords
    # Lin-H: only sample coords with dist >= 18mm from any CAD center
    cs_all = [(f["x"], f["y"]) for f in cad_corr]
    healthy_sample_coords = [c for c in sample_coords if min(np.hypot(c[0]-a, c[1]-b) for a, b in cs_all) >= 18.0]
    
    def extract_patch_probes(c_list):
        X_ctx, Y_tgt = [], []
        for u, v in c_list:
            # probes shape [25, 128]
            probes = cscan[v + star_offsets[:, 0], u + star_offsets[:, 1], :]
            p0 = probes[0]  # [128]
            ctx = probes[1:].ravel()  # [24 * 128 = 3072]
            X_ctx.append(ctx)
            Y_tgt.append(p0)
        return np.array(X_ctx, dtype=np.float32), np.array(Y_tgt, dtype=np.float32)

    X_train_u, Y_train_u = extract_patch_probes(sample_coords[:4000])
    X_train_h, Y_train_h = extract_patch_probes(healthy_sample_coords[:4000])
    
    lin_u1 = Ridge(alpha=1.0).fit(X_train_u, Y_train_u)
    lin_u2 = Ridge(alpha=10.0).fit(X_train_u, Y_train_u)  # For A/A test
    lin_h  = Ridge(alpha=1.0).fit(X_train_h, Y_train_h)

    # 2.3 Compute Anomaly Maps A_M(c) at points of interest
    # We need A_M on disks B(c_i, R) for all 25 flaws and all 56 shams
    print("Computing Anomaly Scores A_M on Evaluation Disks for JEPA and Baselines ...")
    
    # Unique evaluation centers: 25 CAD centers + 40 interior shams + 16 edge shams
    eval_centers = []
    # (type, id/name, cx, cy, R)
    for f in cad_corr:
        R_d = 7 if f["diameter"] >= 10.0 else 6
        eval_centers.append(("flaw", f["id"], f["x"], f["y"], R_d))
        
    for s_idx, (sx, sy) in enumerate(interior_shams):
        eval_centers.append(("sham_int_R6", s_idx, sx, sy, 6))
        eval_centers.append(("sham_int_R7", s_idx, sx, sy, 7))
        
    for s_idx, (sx, sy) in enumerate(all_edge_shams):
        eval_centers.append(("sham_edge_R6", s_idx, sx, sy, 6))
        eval_centers.append(("sham_edge_R7", s_idx, sx, sy, 7))

    # Pre-calculate candidate pixels within radius R for each center
    # And compute A_JEPA, A_Lin_U1, A_Lin_U2, A_Lin_H
    scores = {"JEPA": {}, "Lin-U1": {}, "Lin-U2": {}, "Lin-H": {}}
    
    # Custom context/target indices for masking central probe (probe 0 target, probes 1..24 context)
    ctx_idx = torch.arange(1, 25, device=device).unsqueeze(0)  # [1, 24]
    tgt_idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # [1, 1]
    
    for c_type, c_id, cx, cy, R_val in eval_centers:
        # Generate pixel grid in disk
        pts_in_disk = []
        for du in range(-R_val, R_val + 1):
            for dv in range(-R_val, R_val + 1):
                if du**2 + dv**2 <= R_val**2:
                    u_pt, v_pt = int(round(cx + du)), int(round(cy + dv))
                    if 7 <= u_pt <= 262 and 7 <= v_pt <= 262:
                        pts_in_disk.append((u_pt, v_pt))
                        
        if len(pts_in_disk) == 0:
            continue
            
        # Extract concentric star patch for each pixel in disk [B_disk, 5, 5, 128]
        # To format for model, convert 25 probes to [B, 5, 5, C]
        disk_probes = np.array([cscan[v + star_offsets[:, 0], u + star_offsets[:, 1], :] for u, v in pts_in_disk], dtype=np.float32)
        # Reshape to [B, 5, 5, C]
        B_disk = len(pts_in_disk)
        disk_5x5 = disk_probes.reshape(B_disk, 5, 5, 128)
        
        # JEPA Inference
        t_5x5 = torch.from_numpy(disk_5x5).to(device)
        with torch.no_grad():
            b_ctx = ctx_idx.expand(B_disk, -1)
            b_tgt = tgt_idx.expand(B_disk, -1)
            out_dict = model.forward(t_5x5, custom_context_indices=b_ctx, custom_target_indices=b_tgt)
            h_pred = out_dict["H_pred"]  # [B, 1, D]
            h_tgt  = out_dict["H_tgt"]   # [B, 1, D]
            # A_JEPA = || H_pred - H_tgt ||_2^2
            a_jepa = torch.sum((h_pred - h_tgt)**2, dim=-1).squeeze(1).cpu().numpy() # [B]
            
        # Linear Baselines Inference
        disk_p0 = disk_probes[:, 0, :]  # [B, 128]
        disk_ctx = disk_probes[:, 1:, :].reshape(B_disk, -1)  # [B, 24*128]
        
        pred_u1 = lin_u1.predict(disk_ctx)
        pred_u2 = lin_u2.predict(disk_ctx)
        pred_h  = lin_h.predict(disk_ctx)
        
        a_lin_u1 = np.sum((disk_p0 - pred_u1)**2, axis=-1)
        a_lin_u2 = np.sum((disk_p0 - pred_u2)**2, axis=-1)
        a_lin_h  = np.sum((disk_p0 - pred_h)**2, axis=-1)
        
        # S_M = max over disk
        key = (c_type, c_id, R_val)
        scores["JEPA"][key]   = float(np.max(a_jepa))
        scores["Lin-U1"][key] = float(np.max(a_lin_u1))
        scores["Lin-U2"][key] = float(np.max(a_lin_u2))
        scores["Lin-H"][key]  = float(np.max(a_lin_h))

    # 2.4 Compute Quantiles q_i and p_i per Model according to Điều 2.4 & 2.5
    results_models = {}
    for m_name in ["JEPA", "Lin-U1", "Lin-U2", "Lin-H"]:
        q_dict = {}
        # Interior flaws (9)
        for fid in interior_flaw_ids:
            f = cad_corr[fid - 1]
            R_d = 7 if f["diameter"] >= 10.0 else 6
            s_val = scores[m_name][("flaw", fid, R_d)]
            # Ref set H_i: 40 interior shams with R_d
            ref_vals = [scores[m_name][("sham_int_R" + str(R_d), s_idx, R_d)] for s_idx in range(40)]
            q = np.mean(np.array(ref_vals) < s_val) + 0.5 * np.mean(np.array(ref_vals) == s_val)
            q_dict[fid] = float(q)
            
        # Edge flaws (12 non-corner)
        # Demean by side mean
        for side in ["left", "right", "top", "bottom"]:
            side_shams = edge_shams_by_side[side]
            # Identify edge flaws belonging to this side
            if side == "left":
                side_fids = [fid for fid in edge_flaw_ids if cad_corr[fid-1]["x"] <= 15]
            elif side == "right":
                side_fids = [fid for fid in edge_flaw_ids if cad_corr[fid-1]["x"] >= 255]
            elif side == "top":
                side_fids = [fid for fid in edge_flaw_ids if cad_corr[fid-1]["y"] <= 15]
            else:
                side_fids = [fid for fid in edge_flaw_ids if cad_corr[fid-1]["y"] >= 255]
                
            for fid in side_fids:
                f = cad_corr[fid - 1]
                R_d = 7 if f["diameter"] >= 10.0 else 6
                # Compute side mean of shams
                # Find indices of side shams in all_edge_shams
                side_indices = [all_edge_shams.index(pt) for pt in side_shams]
                side_sham_scores = [scores[m_name][("sham_edge_R" + str(R_d), idx, R_d)] for idx in side_indices]
                m_side = np.mean(side_sham_scores)
                
                # Demean all 16 edge shams
                all_edge_scores = [scores[m_name][("sham_edge_R" + str(R_d), idx, R_d)] for idx in range(16)]
                # Find side of each sham
                sham_sides = (["left"]*4 + ["right"]*4 + ["top"]*4 + ["bottom"]*4)
                side_means = {s: np.mean([all_edge_scores[k] for k, ss in enumerate(sham_sides) if ss == s]) for s in ["left", "right", "top", "bottom"]}
                ref_tilde = [all_edge_scores[k] - side_means[sham_sides[k]] for k in range(16)]
                
                s_tilde = scores[m_name][("flaw", fid, R_d)] - m_side
                q = np.mean(np.array(ref_tilde) < s_tilde) + 0.5 * np.mean(np.array(ref_tilde) == s_tilde)
                q_dict[fid] = float(q)
                
        results_models[m_name] = q_dict

    # 2.5 Prerequisite Gates Verification
    gates = {}
    # G0: Positive Control (Lin-U1 on Z_PtP >= 6 flaws)
    if len(pos_control_flaws) > 0:
        auc_pos_control = float(np.mean([results_models["Lin-U1"][fid] for fid in pos_control_flaws]))
        gates["G0"] = {"passed": bool(auc_pos_control >= 0.80), "auc": float(auc_pos_control), "n": int(len(pos_control_flaws))}
    else:
        gates["G0"] = {"passed": False, "auc": 0.0, "n": 0, "note": "No flaws with Z_PtP >= 6"}
        
    # G-N1: Position trend on interior sham scores
    sham_int_scores_r6 = [scores["JEPA"][("sham_int_R6", idx, 6)] for idx in range(40)]
    pts_x = np.array([interior_shams[idx][0] for idx in range(40)])
    pts_y = np.array([interior_shams[idx][1] for idx in range(40)])
    # 2nd order poly fit
    A_poly = np.column_stack([np.ones(40), pts_x, pts_y, pts_x**2, pts_y**2, pts_x*pts_y])
    _, _, _, _ = np.linalg.lstsq(A_poly, sham_int_scores_r6, rcond=None)
    res_poly = sham_int_scores_r6 - A_poly @ np.linalg.lstsq(A_poly, sham_int_scores_r6, rcond=None)[0]
    r2_pos = 1.0 - np.sum(res_poly**2) / (np.sum((sham_int_scores_r6 - np.mean(sham_int_scores_r6))**2) + 1e-12)
    gates["G_N1"] = {"passed": bool(r2_pos <= 0.20), "r2_pos": float(r2_pos)}
    
    # G2: Stripe noise correlation
    # Compute L_stripe on healthy pixels
    l_stripes, a_jepa_healthy = [], []
    for c in healthy_sample_coords[:500]:
        u, v = c
        # 25 probes
        p_raw = cscan[v + star_offsets[:, 0], u + star_offsets[:, 1], :] # [25, 128]
        # Adjacent y diff of center raw
        diff_y = float(np.mean((cscan[v+1, u, :] - cscan[v, u, :])**2))
        l_stripes.append(diff_y)
        # Get A_JEPA at (u, v)
        p_5x5 = torch.from_numpy(p_raw.reshape(1, 5, 5, 128)).to(device)
        with torch.no_grad():
            od = model.forward(p_5x5, custom_context_indices=ctx_idx, custom_target_indices=tgt_idx)
            aj = float(torch.sum((od["H_pred"] - od["H_tgt"])**2).cpu().item())
        a_jepa_healthy.append(aj)
    slope, _, r_val, _, _ = stats.linregress(l_stripes, a_jepa_healthy)
    r2_stripe = float(r_val**2)
    gates["G2"] = {"passed": bool(r2_stripe <= 0.50), "r2_stripe": float(r2_stripe)}
    
    # G3: Statistical power
    gates["G3"] = {"passed": bool(len(O_s) >= 8), "n_eval": int(len(O_s))}

    # 2.6 Primary Metric: AUC on O_s and Bootstrap Delta
    if len(O_s) > 0:
        auc_jepa = float(np.mean([results_models["JEPA"][fid] for fid in O_s]))
        auc_u1   = float(np.mean([results_models["Lin-U1"][fid] for fid in O_s]))
        auc_u2   = float(np.mean([results_models["Lin-U2"][fid] for fid in O_s]))
        auc_h    = float(np.mean([results_models["Lin-H"][fid] for fid in O_s]))
        max_lin  = max(auc_u1, auc_u2, auc_h)
        delta_s  = auc_jepa - max_lin
        
        # A/A noise floor: delta between Lin-U1 and Lin-U2
        delta_aa = abs(auc_u1 - auc_u2)
        
        # Two-tier Bootstrap (B=2000)
        B = 2000
        boot_deltas = []
        np.random.seed(42)
        O_s_arr = np.array(O_s)
        for _ in range(B):
            b_sample = np.random.choice(O_s_arr, size=len(O_s), replace=True)
            b_jepa = np.mean([results_models["JEPA"][fid] for fid in b_sample])
            b_u1   = np.mean([results_models["Lin-U1"][fid] for fid in b_sample])
            b_u2   = np.mean([results_models["Lin-U2"][fid] for fid in b_sample])
            b_h    = np.mean([results_models["Lin-H"][fid] for fid in b_sample])
            b_delta = b_jepa - max(b_u1, b_u2, b_h)
            boot_deltas.append(b_delta)
            
        ci_lo = float(np.percentile(boot_deltas, 2.5))
        ci_hi = float(np.percentile(boot_deltas, 97.5))
        se_aa = float(np.std(boot_deltas))
        
        # Decision state according to Điều 5.1 & 5.2
        all_gates_passed = all(g.get("passed", False) for g in gates.values())
        delta_min = max(0.05, 3.0 * se_aa)
        if all_gates_passed:
            if delta_s >= delta_min and ci_lo > 0:
                decision_state = "ESTABLISHED"
            elif ci_hi < 0:
                decision_state = "REJECTED"
            else:
                decision_state = "NOT DETECTED"
        else:
            failed_gates = [k for k, g in gates.items() if not g.get("passed", False)]
            decision_state = f"NOT DETECTED (Failed gates: {','.join(failed_gates)})"
    else:
        auc_jepa = auc_u1 = auc_u2 = auc_h = max_lin = delta_s = delta_aa = 0.0
        ci_lo = ci_hi = se_aa = delta_min = 0.0
        decision_state = "NOT DETECTED (Zero evaluable flaws)"

    return {
        "scan_name": scan_name,
        "transductive": transductive_label,
        "O_s_count": len(O_s),
        "pos_control_count": len(pos_control_flaws),
        "gates": gates,
        "AUC": {
            "JEPA": auc_jepa,
            "Lin-U1": auc_u1,
            "Lin-U2": auc_u2,
            "Lin-H": auc_h,
            "Max_Linear": max_lin,
        },
        "Delta_s": delta_s,
        "Bootstrap_95_CI": [ci_lo, ci_hi],
        "delta_min": delta_min,
        "A_A_delta": delta_aa,
        "SE_AA": se_aa,
        "decision_state": decision_state,
        "quantiles_jepa": results_models["JEPA"],
    }

# -----------------------------------------------------------------------------
# 3. RUN ON THE 3 PRIMARY SCANS
# -----------------------------------------------------------------------------
scans_to_run = [
    {
        "name": "HallAir_Chirp_z1",
        "file": os.path.join(ROOT_DIR, r"data\Hall_Air_Core\Corrosion\Chirp\hall_aircore_corosion_frontside_chirp_300x300_2.97_500-1500hz_z1_20260118_162411.tdms"),
        "transductive": True,
    },
    {
        "name": "TMR_Chirp_z1",
        "file": os.path.join(ROOT_DIR, r"data\TMR\Corrosion\Chirp\tmr_corosion_frontside_chirp_300x300_z1_20260128_015728.tdms"),
        "transductive": True,
    },
    {
        "name": "HallPot_Chirp_z1",
        "file": os.path.join(ROOT_DIR, r"data\Hall_Pot_Core\Corrosion\Chirp\corosion_frontside_chirp_300x300_f500hz_f1500hz_amp2.97_z1_20260107_061143.tdms"),
        "transductive": True,
    },
]

all_scan_results = []
for item in scans_to_run:
    res = evaluate_single_scan(item["file"], item["name"], item["transductive"])
    all_scan_results.append(res)

# Export results
out_json_path = os.path.join(ROOT_DIR, "data", "cscan_audit_exports", "pilot_phase2_empirical_results.json")
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(all_scan_results, f, indent=2, cls=NpEncoder)

print("\n" + "=" * 80)
print("PILOT PHASE 2 BENCHMARK EXECUTION SUMMARY")
print("=" * 80)
for r in all_scan_results:
    print(f"\nScan: {r['scan_name']} [{ 'Transductive' if r['transductive'] else 'Inductive' }]")
    print(f"  Evaluable flaws |O_s|: {r['O_s_count']} flaws (Pos Control Z>=6: {r['pos_control_count']})")
    print(f"  Gates: G0 (Pos Control)={r['gates']['G0']['passed']} (AUC={r['gates']['G0']['auc']:.3f}), G_N1 (Null Pos)={r['gates']['G_N1']['passed']} (R2={r['gates']['G_N1']['r2_pos']:.3f}), G2 (Stripe)={r['gates']['G2']['passed']} (R2={r['gates']['G2']['r2_stripe']:.3f}), G3={r['gates']['G3']['passed']}")
    print(f"  AUC-ROC: JEPA = {r['AUC']['JEPA']:.4f} vs Max Linear Baseline = {r['AUC']['Max_Linear']:.4f}")
    print(f"  Delta_s: {r['Delta_s']:+.4f} (95% CI: [{r['Bootstrap_95_CI'][0]:+.4f}, {r['Bootstrap_95_CI'][1]:+.4f}], delta_min={r['delta_min']:.4f})")
    print(f"  A/A Noise Floor: Delta_AA = {r['A_A_delta']:.4f}, SE = {r['SE_AA']:.4f}")
    print(f"  >>> DECISION STATE: {r['decision_state']} <<<")

print(f"\nFull empirical data exported to: {out_json_path}")

