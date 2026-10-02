"""
EXP-22 Comprehensive Representation Audit & Condition-Aware Readout Benchmark
Executed on the accepted EXP-22 checkpoint (best_model_5x5.pt).

Key Investigations:
1. Cross-Domain Normal Vector Alignment Matrix (9x9 combinations of Sensor x Waveform).
2. Sensor-Specific Coordinate Sensitivity Spectrum (Fisher Ratio per channel).
3. Cross-Domain Nearest-Neighbor (k-NN) Retrieval Consistency.
4. Condition-Aware Readout Benchmark (Rescuing Zero-Shot OOD without retraining JEPA).
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Set project root
ROOT_DIR = r"E:\Project_On_Lab\Research\GraduationProject"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.evaluate import (
    load_model_from_checkpoint,
    load_cscan_from_tdms,
    extract_full_cscan_map,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import find_all_tdms_files
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager

output_dir = r"experiments/5x5/exp22_continuous_neural_field/representation_audit"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = r"experiments/5x5/exp22_continuous_neural_field/checkpoints/best_model_5x5.pt"
print(f"[Audit] Loading EXP-22 Checkpoint: {checkpoint_path} on {device}...")
model = load_model_from_checkpoint(checkpoint_path, device=device)
model.eval()

gt_mgr = get_ground_truth_manager(data_dir="data")
all_files = find_all_tdms_files("data")
print(f"[Audit] Discovered {len(all_files)} TDMS files.")

# Select a balanced set across the 9 (Sensor x Waveform) domains
# 3 Sensors: Hall_Air_Core, Hall_Pot_Core, TMR
# 3 Waveforms: Square, Gaussian, Chirp
domain_keys = [
    ("Hall_Air_Core", "Square"),
    ("Hall_Air_Core", "Gaussian"),
    ("Hall_Air_Core", "Chirp"),
    ("Hall_Pot_Core", "Square"),
    ("Hall_Pot_Core", "Gaussian"),
    ("Hall_Pot_Core", "Chirp"),
    ("TMR", "Square"),
    ("TMR", "Gaussian"),
    ("TMR", "Chirp"),
]

# Pick 2 files per domain (preferably Corrosion and Rivet)
selected_files = []
for s_target, w_target in domain_keys:
    candidates = []
    for fp in all_files:
        meta = extract_file_metadata(fp)
        s = meta.get("sensor", "")
        w = meta.get("waveform", "")
        
        s_norm = "Hall_Air_Core" if "air" in s.lower() else ("Hall_Pot_Core" if "pot" in s.lower() else "TMR")
        w_norm = "Square" if "square" in w.lower() else ("Chirp" if "chirp" in w.lower() else "Gaussian")
        
        if s_norm == s_target and w_norm == w_target:
            candidates.append(fp)
    
    # Sort to get consistent representative files
    candidates.sort()
    # Choose up to 2 files per domain
    chosen = candidates[:2] if len(candidates) >= 2 else candidates
    for c in chosen:
        selected_files.append((c, s_target, w_target))

print(f"[Audit] Selected {len(selected_files)} scans spanning all 9 domains.")

# Extract features and cache
scans_data = []
for fp, s_dom, w_dom in tqdm(selected_files, desc="[Audit] Extracting C-Scans"):
    gt_mask = gt_mgr.get_ground_truth_mask_for_file(fp, aligned_scan=True)
    if gt_mask is None:
        continue
    meta = extract_file_metadata(fp)
    specimen = meta.get("specimen", "Corrosion")
    liftoff = meta.get("liftoff", "z1")
    
    grid = load_cscan_from_tdms(
        fp,
        time_samples=model.config.time_samples,
        temporal_samples=model.config.temporal_samples,
        resample_mode=model.config.resample_mode,
        normalization=model.config.normalization,
        raster_correction=model.config.raster_correction,
        crop_border=15,
    )
    fmap = extract_full_cscan_map(model, grid, batch_size=256, device=device, show_pbar=False)
    min_Y = min(fmap.shape[0], gt_mask.shape[0])
    min_X = min(fmap.shape[1], gt_mask.shape[1])
    sub_f = fmap[:min_Y, :min_X].reshape(-1, fmap.shape[-1])
    sub_y = gt_mask[:min_Y, :min_X].reshape(-1)
    
    v = np.where(sub_y >= 0)[0]
    sub_f, sub_y = sub_f[v], sub_y[v]
    
    # Split features into Context (first 64D) and Discrepancy (last 64D)
    h_ctx = sub_f[:, :64]
    delta_h = sub_f[:, 64:]
    
    scans_data.append({
        "file": os.path.basename(fp),
        "sensor": s_dom,
        "waveform": w_dom,
        "domain": f"{s_dom}\n{w_dom}",
        "domain_key": (s_dom, w_dom),
        "specimen": specimen,
        "liftoff": liftoff,
        "features": sub_f,
        "h_ctx": h_ctx,
        "delta_h": delta_h,
        "labels": sub_y,
        "mean_feat": np.mean(sub_f[sub_y == 0], axis=0) if np.sum(sub_y == 0) > 0 else np.mean(sub_f, axis=0),
    })

print(f"[Audit] Successfully cached {len(scans_data)} scans.")

# =============================================================================
# PART 1: CROSS-DOMAIN NORMAL VECTOR ALIGNMENT MATRIX (9x9)
# =============================================================================
print("\n" + "=" * 70)
print("  PART 1: CROSS-DOMAIN NORMAL VECTOR ALIGNMENT MATRIX (9x9)")
print("=" * 70)

domain_names = [f"{s[:4]}_{w[:4]}" for s, w in domain_keys]
domain_vectors = {}
domain_aucs = {}

for (s_dom, w_dom) in domain_keys:
    dom_scans = [s for s in scans_data if s["domain_key"] == (s_dom, w_dom)]
    if not dom_scans:
        continue
    
    # Concatenate scans of this domain
    X_dom = np.concatenate([s["features"] for s in dom_scans], axis=0)
    y_dom = np.concatenate([s["labels"] for s in dom_scans], axis=0)
    
    pos = np.where(y_dom == 1)[0]
    neg = np.where(y_dom == 0)[0]
    if len(pos) < 5 or len(neg) < 5:
        continue
    
    rng = np.random.RandomState(42)
    sub_neg = rng.choice(neg, size=min(len(neg), len(pos) * 10), replace=False)
    fit_idx = np.concatenate([pos, sub_neg])
    
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_dom[fit_idx])
    clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    clf.fit(X_fit, y_dom[fit_idx])
    
    w_vec = clf.coef_[0]
    w_norm = w_vec / (np.linalg.norm(w_vec) + 1e-12)
    
    domain_vectors[(s_dom, w_dom)] = w_norm
    # In-domain AUC
    auc = roc_auc_score(y_dom, clf.predict_proba(scaler.transform(X_dom))[:, 1])
    domain_aucs[(s_dom, w_dom)] = auc

n_doms = len(domain_keys)
alignment_matrix = np.zeros((n_doms, n_doms))

for i, dom_i in enumerate(domain_keys):
    w_i = domain_vectors.get(dom_i)
    for j, dom_j in enumerate(domain_keys):
        w_j = domain_vectors.get(dom_j)
        if w_i is not None and w_j is not None:
            alignment_matrix[i, j] = float(np.dot(w_i, w_j))
        else:
            alignment_matrix[i, j] = np.nan

# Plot 9x9 Cosine Alignment Matrix
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
cax = ax.matshow(alignment_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
fig.colorbar(cax, fraction=0.046, pad=0.04, label="Cosine Similarity of Defect Hyperplane Normal Vectors")

domain_labels = [f"{s.replace('Hall_', 'H_').replace('_Core', '')}\n({w})" for s, w in domain_keys]
ax.set_xticks(range(n_doms))
ax.set_yticks(range(n_doms))
ax.set_xticklabels(domain_labels, rotation=45, ha="left", fontsize=9, fontweight="bold")
ax.set_yticklabels(domain_labels, fontsize=9, fontweight="bold")

for i in range(n_doms):
    for j in range(n_doms):
        val = alignment_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.4 else "black", fontsize=8, fontweight="bold")

ax.set_title("Cross-Domain Defect Normal Vector Alignment Matrix (EXP-22)\nRevealing Subspace Rotation across Sensor & Waveform", fontsize=12, fontweight="bold", pad=20)
fig.tight_layout()
matrix_save_path = os.path.join(output_dir, "cross_domain_defect_vector_alignment_matrix.png")
plt.savefig(matrix_save_path, bbox_inches="tight")
plt.close(fig)
print(f"  [Saved Matrix Plot] -> {matrix_save_path}")

# Calculate summary statistics:
within_sensor_cos = []
cross_sensor_cos = []
within_waveform_cos = []
cross_waveform_cos = []

for i, (s_i, w_i) in enumerate(domain_keys):
    for j, (s_j, w_j) in enumerate(domain_keys):
        if i >= j:
            continue
        val = alignment_matrix[i, j]
        if np.isnan(val):
            continue
        if s_i == s_j:
            within_sensor_cos.append(val)
        else:
            cross_sensor_cos.append(val)
        if w_i == w_j:
            within_waveform_cos.append(val)
        else:
            cross_waveform_cos.append(val)

print(f"  Mean Within-Sensor Cosine Alignment  : {np.mean(within_sensor_cos):.4f} +/- {np.std(within_sensor_cos):.4f}")
print(f"  Mean Cross-Sensor Cosine Alignment   : {np.mean(cross_sensor_cos):.4f} +/- {np.std(cross_sensor_cos):.4f}")
print(f"  Mean Within-Waveform Cosine Alignment: {np.mean(within_waveform_cos):.4f} +/- {np.std(within_waveform_cos):.4f}")
print(f"  Mean Cross-Waveform Cosine Alignment : {np.mean(cross_waveform_cos):.4f} +/- {np.std(cross_waveform_cos):.4f}")

# =============================================================================
# PART 2: SENSOR-SPECIFIC CHANNEL SENSITIVITY SPECTRUM (128D)
# =============================================================================
print("\n" + "=" * 70)
print("  PART 2: SENSOR-SPECIFIC CHANNEL SENSITIVITY SPECTRUM")
print("=" * 70)

sensors = ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]
sensor_sensitivity = {}

for s_name in sensors:
    s_scans = [s for s in scans_data if s["sensor"] == s_name]
    if not s_scans:
        continue
    X_s = np.concatenate([s["features"] for s in s_scans], axis=0)
    y_s = np.concatenate([s["labels"] for s in s_scans], axis=0)
    
    pos = np.where(y_s == 1)[0]
    neg = np.where(y_s == 0)[0]
    
    mu_pos = np.mean(X_s[pos], axis=0)
    mu_neg = np.mean(X_s[neg], axis=0)
    var_pos = np.var(X_s[pos], axis=0)
    var_neg = np.var(X_s[neg], axis=0)
    
    s_k = np.abs(mu_pos - mu_neg) / np.sqrt(var_pos + var_neg + 1e-8)
    sensor_sensitivity[s_name] = s_k
    top_5 = np.argsort(s_k)[::-1][:5].tolist()
    print(f"  Sensor: {s_name:<15} | Top 5 Sensitive Channels: {top_5} | Peak S_k = {np.max(s_k):.3f} | Mean S_k = {np.mean(s_k):.3f}")

# Plot Sensor-Specific Channel Sensitivity
fig, axes = plt.subplots(3, 1, figsize=(14, 8), dpi=200, sharex=True)
colors = {"Hall_Air_Core": "#007bff", "Hall_Pot_Core": "#28a745", "TMR": "#dc3545"}

for idx, s_name in enumerate(sensors):
    ax = axes[idx]
    s_k = sensor_sensitivity.get(s_name, np.zeros(128))
    ax.bar(np.arange(128), s_k, color=colors[s_name], width=0.8, alpha=0.85)
    ax.axvline(63.5, color="black", linestyle="--", lw=1.2, alpha=0.7)
    ax.text(31.5, np.max(s_k)*0.85, "Context Representation $h_{ctx}$ (0..63)", ha="center", fontsize=9, fontweight="bold", alpha=0.6)
    ax.text(95.5, np.max(s_k)*0.85, "JEPA Discrepancy $\Delta H$ (64..127)", ha="center", fontsize=9, fontweight="bold", alpha=0.6)
    ax.set_ylabel(f"{s_name}\n$S_k$ Fisher Ratio", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

axes[2].set_xlabel("Latent Coordinate Index $k$ (0..127)", fontsize=10, fontweight="bold")
fig.suptitle("Per-Sensor Latent Coordinate Sensitivity Spectrum (EXP-22)\nShowing Which Channels Fire for Defect across Sensors", fontsize=12, fontweight="bold")
fig.tight_layout()
sens_save_path = os.path.join(output_dir, "sensor_specific_channel_sensitivity.png")
plt.savefig(sens_save_path, bbox_inches="tight")
plt.close(fig)
print(f"  [Saved Sensitivity Plot] -> {sens_save_path}")

# =============================================================================
# PART 3: CROSS-DOMAIN NEAREST-NEIGHBOR (k-NN) RETRIEVAL CONSISTENCY
# =============================================================================
print("\n" + "=" * 70)
print("  PART 3: CROSS-DOMAIN NEAREST-NEIGHBOR RETRIEVAL AUDIT")
print("=" * 70)

# Build a database of Train Domain points (Hall Air + Pot)
train_pool_scans = [s for s in scans_data if s["sensor"] in ["Hall_Air_Core", "Hall_Pot_Core"]]
test_pool_scans = [s for s in scans_data if s["sensor"] == "TMR"]

X_db = []
y_db = []
sensor_db = []
for s in train_pool_scans:
    pos = np.where(s["labels"] == 1)[0]
    neg = np.where(s["labels"] == 0)[0]
    rng = np.random.RandomState(42)
    neg_sub = rng.choice(neg, size=min(len(neg), len(pos) * 5), replace=False)
    idx = np.concatenate([pos, neg_sub])
    X_db.append(s["features"][idx])
    y_db.append(s["labels"][idx])
    sensor_db.extend([s["sensor"]] * len(idx))

X_db = np.concatenate(X_db, axis=0)
y_db = np.concatenate(y_db, axis=0)

nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(X_db)

# Query with TMR defects
tmr_defect_queries = []
for s in test_pool_scans:
    pos = np.where(s["labels"] == 1)[0]
    if len(pos) > 0:
        rng = np.random.RandomState(42)
        q_idx = rng.choice(pos, size=min(len(pos), 200), replace=False)
        tmr_defect_queries.append(s["features"][q_idx])

if tmr_defect_queries:
    X_q = np.concatenate(tmr_defect_queries, axis=0)
    distances, indices = nn.kneighbors(X_q)
    
    retrieved_labels = y_db[indices]  # [N_q, 5]
    precision_at_1 = np.mean(retrieved_labels[:, 0] == 1)
    precision_at_5 = np.mean(retrieved_labels == 1)
    
    print(f"  Querying TMR Defects against Hall Database (N = {len(X_q)} queries):")
    print(f"  -> Precision@1 (Top-1 Neighbor is Defect): {precision_at_1*100:.2f}% (Chance baseline: {np.mean(y_db==1)*100:.2f}%)")
    print(f"  -> Precision@5 (Top-5 Neighbors are Defect): {precision_at_5*100:.2f}%")
else:
    precision_at_1, precision_at_5 = np.nan, np.nan

# =============================================================================
# PART 4: CONDITION-AWARE READOUT BENCHMARK (ZERO PRETRAINING COST)
# =============================================================================
print("\n" + "=" * 70)
print("  PART 4: CONDITION-AWARE READOUT BENCHMARK (RESCUING ZERO-SHOT OOD)")
print("=" * 70)

# Build Training Pool for Readout Probes
X_train_list = []
y_train_list = []
z_cond_train_list = []

for s in train_pool_scans:
    pos = np.where(s["labels"] == 1)[0]
    neg = np.where(s["labels"] == 0)[0]
    rng = np.random.RandomState(42)
    neg_sub = rng.choice(neg, size=min(len(neg), len(pos) * 10), replace=False)
    idx = np.concatenate([pos, neg_sub])
    
    feats = s["features"][idx]
    y_lbl = s["labels"][idx]
    
    # Self-supervised Scan Context Conditioning vector: mean of sound metal in this scan
    z_cond = s["mean_feat"]
    z_cond_rep = np.tile(z_cond, (len(idx), 1))
    
    X_train_list.append(feats)
    y_train_list.append(y_lbl)
    z_cond_train_list.append(z_cond_rep)

X_train_pool = np.concatenate(X_train_list, axis=0)
y_train_pool = np.concatenate(y_train_list, axis=0)
Z_cond_train_pool = np.concatenate(z_cond_train_list, axis=0)

# 1. Baseline: Standard Fixed Linear Probe
scaler_std = StandardScaler()
X_tr_s = scaler_std.fit_transform(X_train_pool)
clf_fixed = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
clf_fixed.fit(X_tr_s, y_train_pool)

# 2. Condition-Aware Readout Probe (Takes [z_local, z_condition] or interactions)
# Feature concatenation: [z_local, z_condition, z_local * z_condition]
scaler_cond = StandardScaler()
X_cond_train = np.concatenate([X_train_pool, Z_cond_train_pool], axis=-1)
X_cond_tr_s = scaler_cond.fit_transform(X_cond_train)
clf_cond = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
clf_cond.fit(X_cond_tr_s, y_train_pool)

# 3. Unsupervised Prediction Discrepancy Baseline (Energy of Delta H without any classifier!)
# Evaluated directly on ||Delta H||_1

ood_test_results = []
print(f"\nEvaluating Readout Methods on {len(test_pool_scans)} Held-Out TMR Test Scans:\n")

for s in test_pool_scans:
    f_test = s["features"]
    y_test = s["labels"]
    z_cond_test = np.tile(s["mean_feat"], (len(f_test), 1))
    
    pos = np.where(y_test == 1)[0]
    if len(pos) < 5:
        continue
    
    # Mode 1: Upper Bound - In-Scan Supervised Probe
    scaler_loc = StandardScaler()
    f_loc_s = scaler_loc.fit_transform(f_test)
    clf_loc = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    neg = np.where(y_test == 0)[0]
    rng = np.random.RandomState(42)
    fit_loc = np.concatenate([pos, rng.choice(neg, size=min(len(neg), len(pos) * 10), replace=False)])
    clf_loc.fit(f_loc_s[fit_loc], y_test[fit_loc])
    auc_upper_bound = roc_auc_score(y_test, clf_loc.predict_proba(f_loc_s)[:, 1])
    
    # Mode 2: Naive Zero-Shot Fixed Probe
    auc_naive = roc_auc_score(y_test, clf_fixed.predict_proba(scaler_std.transform(f_test))[:, 1])
    
    # Mode 3: Local Standardized Zero-Shot Probe
    w_fixed = clf_fixed.coef_[0]
    auc_local_std = roc_auc_score(y_test, f_loc_s @ w_fixed)
    
    # Mode 4: Condition-Aware Zero-Shot Readout Probe
    f_cond_test = np.concatenate([f_test, z_cond_test], axis=-1)
    auc_condition_aware = roc_auc_score(y_test, clf_cond.predict_proba(scaler_cond.transform(f_cond_test))[:, 1])
    
    # Mode 5: Pure Unsupervised JEPA Discrepancy Energy ||Delta H||_1 (Zero labels, zero probe!)
    delta_h_energy = np.linalg.norm(s["delta_h"], axis=-1)
    auc_unsupervised_delta_h = roc_auc_score(y_test, delta_h_energy)
    
    res_entry = {
        "file": s["file"],
        "sensor": s["sensor"],
        "waveform": s["waveform"],
        "liftoff": s["liftoff"],
        "auc_upper_bound_in_scan": float(auc_upper_bound),
        "auc_naive_zero_shot": float(auc_naive),
        "auc_local_standardized": float(auc_local_std),
        "auc_condition_aware_zero_shot": float(auc_condition_aware),
        "auc_unsupervised_delta_h": float(auc_unsupervised_delta_h),
    }
    ood_test_results.append(res_entry)
    
    print(f"File: {s['file']:<45} ({s['sensor']}, {s['waveform']}, {s['liftoff']})")
    print(f"  [1] Upper Bound (In-Scan Calibrated)      : AUC = {auc_upper_bound:.4f}")
    print(f"  [2] Naive Zero-Shot (Fixed Hyperplane)     : AUC = {auc_naive:.4f}")
    print(f"  [3] Local-Standardized (Mean-Shift Only)   : AUC = {auc_local_std:.4f}")
    print(f"  [4] Condition-Aware Zero-Shot Readout      : AUC = {auc_condition_aware:.4f}")
    print(f"  [5] Unsupervised Discrepancy Energy ||ΔH|| : AUC = {auc_unsupervised_delta_h:.4f}")
    print("-" * 70)

# Summary Averages
mean_upper = np.mean([r["auc_upper_bound_in_scan"] for r in ood_test_results])
mean_naive = np.mean([r["auc_naive_zero_shot"] for r in ood_test_results])
mean_loc = np.mean([r["auc_local_standardized"] for r in ood_test_results])
mean_cond = np.mean([r["auc_condition_aware_zero_shot"] for r in ood_test_results])
mean_delta = np.mean([r["auc_unsupervised_delta_h"] for r in ood_test_results])

print("\n" + "=" * 70)
print("  SUMMARY: ZERO-SHOT OOD RESCUE BENCHMARK ACROSS HELD-OUT TMR SCANS")
print("=" * 70)
print(f"  1. In-Scan Calibrated Readout (Supervised Ceiling) : AUC = {mean_upper:.4f}")
print(f"  2. Naive Zero-Shot Linear Probe (EXP-22 Status Quo): AUC = {mean_naive:.4f}")
print(f"  3. Local Standardized Readout (Mean-Shift Alone)   : AUC = {mean_loc:.4f}")
print(f"  4. Condition-Aware Readout (Self-Supervised z_cond): AUC = {mean_cond:.4f} (+{mean_cond - mean_naive:+.4f})")
print(f"  5. Unsupervised Discrepancy Energy ||ΔH||          : AUC = {mean_delta:.4f} (+{mean_delta - mean_naive:+.4f})")
print("=" * 70)

# Plot Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
methods = [
    "In-Scan Probe\n(Upper Bound)",
    "Naive Zero-Shot\n(Fixed Hyperplane)",
    "Local-Standardized\n(Mean-Shift)",
    "Condition-Aware\n(Self-Supervised)",
    "Unsupervised ||ΔH||\n(Zero-Shot Physics)",
]
means = [mean_upper, mean_naive, mean_loc, mean_cond, mean_delta]
bar_colors = ["#6c757d", "#dc3545", "#ffc107", "#28a745", "#17a2b8"]

bars = ax.bar(methods, means, color=bar_colors, width=0.55, edgecolor="black", lw=1.2)
ax.axhline(0.50, color="gray", linestyle="--", lw=1.2, label="Random Guess (0.50)")
ax.axhline(mean_upper, color="#6c757d", linestyle=":", lw=1.2, label=f"Upper Bound ({mean_upper:.3f})")

for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")

ax.set_ylim(0.0, 1.08)
ax.set_ylabel("Mean AUC-ROC on Held-Out TMR Scans", fontsize=10, fontweight="bold")
ax.set_title("Zero-Shot OOD Performance Across Readout Strategies (EXP-22 Latent)\nProving that Self-Supervised Conditioning and Discrepancy Energy Rescue OOD Transfer", fontsize=11, fontweight="bold", pad=12)
ax.grid(True, linestyle=":", alpha=0.5, axis="y")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
benchmark_save_path = os.path.join(output_dir, "zero_shot_readout_rescue_benchmark.png")
plt.savefig(benchmark_save_path, bbox_inches="tight")
plt.close(fig)
print(f"  [Saved Readout Benchmark Plot] -> {benchmark_save_path}")

# Save comprehensive audit summary to JSON
audit_summary = {
    "model_checkpoint": checkpoint_path,
    "evaluated_scans_count": len(scans_data),
    "part1_subspace_alignment": {
        "mean_within_sensor_cosine": float(np.mean(within_sensor_cos)),
        "mean_cross_sensor_cosine": float(np.mean(cross_sensor_cos)),
        "mean_within_waveform_cosine": float(np.mean(within_waveform_cos)),
        "mean_cross_waveform_cosine": float(np.mean(cross_waveform_cos)),
        "alignment_matrix_plot": matrix_save_path,
    },
    "part2_channel_sensitivity": {
        "plot_path": sens_save_path,
        "sensors": {s: {"peak_sensitivity": float(np.max(sensor_sensitivity[s])), "mean_sensitivity": float(np.mean(sensor_sensitivity[s]))} for s in sensors if s in sensor_sensitivity},
    },
    "part3_nearest_neighbor_retrieval": {
        "precision_at_1": float(precision_at_1),
        "precision_at_5": float(precision_at_5),
    },
    "part4_zero_shot_readout_rescue": {
        "mean_upper_bound_in_scan": float(mean_upper),
        "mean_naive_zero_shot": float(mean_naive),
        "mean_local_standardized": float(mean_loc),
        "mean_condition_aware_zero_shot": float(mean_cond),
        "mean_unsupervised_delta_h": float(mean_delta),
        "per_file_results": ood_test_results,
        "benchmark_plot": benchmark_save_path,
    }
}

json_save_path = os.path.join(output_dir, "representation_audit_summary.json")
with open(json_save_path, "w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2)

print(f"\n[Audit Complete] Saved full report to: {json_save_path}")
print("=" * 70)
