"""
Latent-subspace audit for Cross-Condition PECT-JEPA (no retraining).

Tests the four design claims DIRECTLY on frozen latents (not via readouts):
  Claim 1 (roles):        z_c carries condition/background, z_s carries defects.
  Claim 2 (invariance):   z_s(A) ~= z_s(B) same coordinate, different condition.
  Claim 3 (OOD):          Claims 1-2 still hold on unseen sensor/waveform.
  Claim 4 (grounding):    pooled z_c predicts the condition token / scan id.

Method: per-probe H_all (tokenizer + context encoder, no pooling loss) split
into [z_c | z_s], pooled center+mean like the downstream readout.
  T1 variance/decomposition per subspace (sound/defect/total, |mu_d-mu_s|)
  T2 subspace probes (LR on z_c-only / z_s-only / full), in-domain + OOD
  T3 A/B same-coordinate invariance (cosine + normalized MSE per subspace)
  T4 energy audit (||z||^2 per subspace, sound vs defect, in vs OOD)
  T5 scan-id decodability per subspace (grounding check, in-domain pool)
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
    _iter_with_prefetch,
)
from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args():
    parser = argparse.ArgumentParser(description="z_c / z_s mechanistic audit (no retraining)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--prefetch", type=lambda v: v.lower() == "true", default=True)
    parser.add_argument("--n_coords", type=int, default=2000)
    return parser.parse_args()


@torch.inference_mode()
def extract_subspaces(model, cscan: np.ndarray, device, batch_size: int = 8192):
    """Full-scan H_all split into pooled carrier/scattering latents.

    Returns zc [H, W, 2*D_c], zs [H, W, 2*D_s] with center+mean pooling,
    mirroring CrossConditionJEPA.extract_features ('scattering'/'carrier').
    """
    from src.PECT_JEPA.cross_condition.data.topologies import ConcentricStarTopology
    cfg = model.config
    topo = ConcentricStarTopology(star_radii=tuple(getattr(cfg, "star_radii", (1, 3, 7))))
    offsets = topo.offsets
    H, W, C = cscan.shape
    pad = int(max(1, np.max(np.abs(offsets))))
    padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rr_flat, cc_flat = (rr.flatten() + pad), (cc.flatten() + pad)
    N = len(rr_flat)
    orr, occ = offsets[:, 0], offsets[:, 1]
    D_c = getattr(cfg, "carrier_dim", cfg.embed_dim // 2)
    use_amp = getattr(cfg, "mixed_precision", True)
    zc_out, zs_out = [], []
    model.eval()
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        b_r, b_c = rr_flat[s:e], cc_flat[s:e]
        patch = padded[b_r[:, None] + orr[None, :], b_c[:, None] + occ[None, :]]
        patch = patch.reshape(e - s, 5, 5, C)
        x = torch.from_numpy(patch).pin_memory().to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            tokens, pos = model.tokenizer(x)
            h_all = model.context_encoder(tokens, pos)
        h_c, h_s = h_all[..., :D_c], h_all[..., D_c:]
        zc_out.append(torch.cat([h_c[:, 0], h_c[:, 1:].mean(1)], -1).cpu().float().numpy())
        zs_out.append(torch.cat([h_s[:, 0], h_s[:, 1:].mean(1)], -1).cpu().float().numpy())
    zc = np.concatenate(zc_out, 0).reshape(H, W, -1)
    zs = np.concatenate(zs_out, 0).reshape(H, W, -1)
    return zc, zs


def find(all_files, specimen=None, sensor=None, waveform=None, liftoff=None):
    out = []
    for fp in all_files:
        m = extract_file_metadata(fp)
        if specimen and m["specimen"] != specimen:
            continue
        if sensor and m["sensor"] != sensor:
            continue
        if waveform and m["waveform"] != waveform:
            continue
        if liftoff and m["liftoff"] != liftoff:
            continue
        out.append(fp)
    return sorted(out)


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path, eval_out = resolve_paths(args)
    out_dir = args.out_dir or os.path.join(eval_out, "latent_audit")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_config = ckpt.get("config", None)
    config = raw_config if isinstance(raw_config, CrossConditionConfig) else CrossConditionConfig()
    model = CrossConditionJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    assert getattr(config, "latent_disentanglement", False), "ckpt is not disentangled"
    D_c = getattr(config, "carrier_dim", config.embed_dim // 2)
    print(f"Disentangled ckpt: D_c={D_c} D_s={config.embed_dim - D_c}", flush=True)

    gt_masks = load_ground_truth_masks()
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]
    splits = get_canonical_train_val_test_splits(all_files, val_files_count=4, seed=42)
    get_canonical_transformer()

    # ---- file selection: in-domain pair + OOD probes ----
    baseA = find(all_files, "Corrosion", "Hall_Air_Core", "Gaussian", "z1")[:1]
    baseB = find(all_files, "Corrosion", "Hall_Pot_Core", "Gaussian", "z1")[:1]
    ood_sensor = find(all_files, "Corrosion", "TMR", "Gaussian", "z1")[:1]
    ood_wave = find(all_files, "Corrosion", "Hall_Pot_Core", "Chirp", "z1")
    if not ood_wave:
        ood_wave = find(all_files, "Corrosion", None, "Chirp", None)[:1]
    else:
        ood_wave = ood_wave[:1]
    ood_spec = find(all_files, "Rivet_v2", "Hall_Air_Core", "Square", None)[:1]
    sel = {"baseA": baseA, "baseB": baseB, "ood_sensor": ood_sensor,
           "ood_wave": ood_wave, "ood_spec": ood_spec}
    for k, v in sel.items():
        print(f"  {k}: {v[0] if v else 'MISSING'}", flush=True)
    needed = sorted({f for v in sel.values() for f in v})

    # ---- train pool (subspace + full features, scan ids kept) ----
    train_files = [fp for fp in splits["train"]
                   if extract_file_metadata(fp)["specimen"] in gt_masks]
    pool_need = sorted(set(train_files) | set(needed))
    store: Dict[str, Any] = {}
    for fp, cscan in tqdm(_iter_with_prefetch(pool_need, args.prefetch),
                          total=len(pool_need), desc="load+extract", dynamic_ncols=True):
        sp = extract_file_metadata(fp)["specimen"]
        zc, zs = extract_subspaces(model, cscan, device, args.batch_size)
        store[fp] = {"zc": zc, "zs": zs, "gt": gt_masks[sp]["mask"]}

    rng = np.random.RandomState(42)
    PX_c, PX_s, PX_f, PY, PS = [], [], [], [], []
    for fp in train_files:
        d = store[fp]
        fz_c = d["zc"].reshape(-1, d["zc"].shape[-1]).astype(np.float32)
        fz_s = d["zs"].reshape(-1, d["zs"].shape[-1]).astype(np.float32)
        fy = d["gt"].reshape(-1).astype(np.int64)
        pos = np.where(fy == 1)[0]
        neg = np.where(fy == 0)[0]
        sub_n = min(len(neg), max(len(pos) * 4, 2500))
        sel_idx = np.concatenate([pos, rng.choice(neg, size=sub_n, replace=False)])
        PX_c.append(fz_c[sel_idx]); PX_s.append(fz_s[sel_idx])
        PX_f.append(np.concatenate([fz_c[sel_idx], fz_s[sel_idx]], axis=1))
        PY.append(fy[sel_idx]); PS += [fp] * len(sel_idx)
    Xc, Xs, Xf = np.concatenate(PX_c, 0), np.concatenate(PX_s, 0), np.concatenate(PX_f, 0)
    Y = np.concatenate(PY, 0)
    print(f"Pool: {len(Y)} ({np.sum(Y==1)} defect)", flush=True)

    def fit_probe(X, y):
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=1.0, max_iter=500, tol=1e-3, class_weight="balanced",
                                random_state=42, solver="lbfgs").fit(sc.transform(X), y)
        return sc, lr

    sc_c, lr_c = fit_probe(Xc, Y)
    sc_s, lr_s = fit_probe(Xs, Y)
    sc_f, lr_f = fit_probe(Xf, Y)

    def auc_of(lr, sc, X, y):
        p = lr.predict_proba(sc.transform(X))[:, 1]
        return (float(roc_auc_score(y, p)), float(average_precision_score(y, p))) \
            if len(np.unique(y)) > 1 else (0.0, 0.0)

    # ================= T1+T4: decomposition + energy =================
    print("\n==== T1 decomposition | T4 energy (sound_var, |mu_d-mu_s|, E=mean||z||^2) ====", flush=True)
    t1_rows = []
    for key in ["baseA", "baseB", "ood_sensor", "ood_wave", "ood_spec"] + ["train_pool"]:
        if key == "train_pool":
            zc = np.concatenate([store[fp]["zc"].reshape(-1, store[fp]["zc"].shape[-1]) for fp in train_files], 0)
            zs = np.concatenate([store[fp]["zs"].reshape(-1, store[fp]["zs"].shape[-1]) for fp in train_files], 0)
            gt = np.concatenate([store[fp]["gt"].reshape(-1) for fp in train_files], 0)
            name = "train_pool(12)"
        else:
            if not sel[key]:
                continue
            fp = sel[key][0]
            d = store[fp]
            zc, zs = d["zc"].reshape(-1, d["zc"].shape[-1]), d["zs"].reshape(-1, d["zs"].shape[-1])
            gt = d["gt"].reshape(-1)
            name = f"{key}:{extract_file_metadata(fp)['sensor']}/{extract_file_metadata(fp)['waveform']}"
        m = gt > 0.5
        row = {"scan": name}
        for tag, z in [("zc", zc), ("zs", zs)]:
            zsnd, zdef = z[~m], z[m]
            row[f"{tag}_sound_var"] = round(float(np.mean(np.var(zsnd, axis=0))), 6)
            row[f"{tag}_gap"] = round(float(np.linalg.norm(zdef.mean(0) - zsnd.mean(0))), 4)
            row[f"{tag}_E_snd"] = round(float(np.mean(np.sum(zsnd ** 2, axis=1))), 3)
            row[f"{tag}_E_def"] = round(float(np.mean(np.sum(zdef ** 2, axis=1))), 3)
        t1_rows.append(row)
        print(f"{row['scan']:42s} zc[var={row['zc_sound_var']:.5f} gap={row['zc_gap']:.3f} E={row['zc_E_snd']:.2f}/{row['zc_E_def']:.2f}] "
              f"zs[var={row['zs_sound_var']:.5f} gap={row['zs_gap']:.3f} E={row['zs_E_snd']:.2f}/{row['zs_E_def']:.2f}]", flush=True)

    # ================= T2: subspace probes =================
    print("\n==== T2 subspace probes (AUC/AP: zc-only | zs-only | full) ====", flush=True)
    t2_rows = []
    eval_keys = ["baseA", "baseB", "ood_sensor", "ood_wave", "ood_spec"]
    for key in eval_keys:
        if not sel[key]:
            continue
        fp = sel[key][0]
        d = store[fp]
        fz_c = d["zc"].reshape(-1, d["zc"].shape[-1]).astype(np.float32)
        fz_s = d["zs"].reshape(-1, d["zs"].shape[-1]).astype(np.float32)
        y = d["gt"].reshape(-1).astype(np.int64)
        v = y >= 0
        yv = y[v]
        a_c = auc_of(lr_c, sc_c, fz_c[v], yv)
        a_s = auc_of(lr_s, sc_s, fz_s[v], yv)
        a_f = auc_of(lr_f, sc_f, np.concatenate([fz_c[v], fz_s[v]], 1), yv)
        meta = extract_file_metadata(fp)
        row = {"scan": f"{key}:{meta['sensor']}/{meta['waveform']}/{meta['specimen']}",
               "zc_auc": round(a_c[0], 4), "zc_ap": round(a_c[1], 4),
               "zs_auc": round(a_s[0], 4), "zs_ap": round(a_s[1], 4),
               "full_auc": round(a_f[0], 4), "full_ap": round(a_f[1], 4)}
        t2_rows.append(row)
        print(f"{row['scan']:48s} zc={row['zc_auc']:.3f}/{row['zc_ap']:.3f} zs={row['zs_auc']:.3f}/{row['zs_ap']:.3f} full={row['full_auc']:.3f}/{row['full_ap']:.3f}", flush=True)

    # ================= T3: A/B same-coordinate invariance =================
    print("\n==== T3 same-coordinate A/B invariance (cos sim + normMSE zs | zc) ====", flush=True)
    pairs = [("baseA", "baseB", "in-domain sensor shift")]
    if sel["baseA"] and sel["ood_sensor"]:
        pairs.append(("baseA", "ood_sensor", "OOD sensor shift"))
    if sel["baseA"] and sel["ood_wave"]:
        pairs.append(("baseA", "ood_wave", "OOD waveform shift"))
    t3_rows = []
    for ka, kb, label in pairs:
        A, B = store[sel[ka][0]], store[sel[kb][0]]
        Hh, Ww = A["zc"].shape[:2]
        rr = rng.randint(0, Hh, size=args.n_coords)
        cc = rng.randint(0, Ww, size=args.n_coords)
        row = {"pair": label}
        for tag in ["zc", "zs"]:
            a = A[tag][rr, cc].astype(np.float64)
            b = B[tag][rr, cc].astype(np.float64)
            cos = float(np.mean(np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)))
            nmse = float(np.mean(np.sum((a - b) ** 2, -1) / (np.mean(np.sum(a ** 2, -1)) + np.mean(np.sum(b ** 2, -1)) + 1e-12)))
            row[f"{tag}_cos"] = round(cos, 4)
            row[f"{tag}_nmse"] = round(nmse, 4)
        t3_rows.append(row)
        print(f"{label:24s} zc[cos={row['zc_cos']:.3f} nmse={row['zc_nmse']:.3f}] zs[cos={row['zs_cos']:.3f} nmse={row['zs_nmse']:.3f}]", flush=True)

    # ================= T5: scan-id decodability (grounding) =================
    print("\n==== T5 scan-id classification accuracy (zc | zs | full) ====", flush=True)
    from sklearn.linear_model import LogisticRegression as LR2
    ids = sorted(set(PS))
    idmap = {f: i for i, f in enumerate(ids)}
    Yid = np.array([idmap[f] for f in PS])
    acc = {}
    for tag, X in [("zc", Xc), ("zs", Xs), ("full", Xf)]:
        sc = StandardScaler().fit(X)
        clf = LR2(C=1.0, max_iter=300, random_state=42).fit(sc.transform(X), Yid)
        acc[tag] = round(float(clf.score(sc.transform(X), Yid)), 4)
    print(f"scan-id acc: zc={acc['zc']} zs={acc['zs']} full={acc['full']} (chance={1/len(ids):.3f})", flush=True)

    # ---- save ----
    with open(os.path.join(out_dir, "t1_decomposition.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t1_rows[0].keys()))
        w.writeheader(); w.writerows(t1_rows)
    with open(os.path.join(out_dir, "t2_subspace_probes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t2_rows[0].keys()))
        w.writeheader(); w.writerows(t2_rows)
    with open(os.path.join(out_dir, "t3_invariance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t3_rows[0].keys()))
        w.writeheader(); w.writerows(t3_rows)
    with open(os.path.join(out_dir, "t5_scanid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subspace", "acc"])
        w.writeheader(); w.writerows([{"subspace": k, "acc": v} for k, v in acc.items()])
    print(f"Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
