"""Source-only diagnostics. Never used to select the campaign checkpoint."""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.paired_dataset import corner_median_ref_patch
from ..data.split import extract_file_metadata
from ..data.topologies import ConcentricStarTopology
from .common import digest, rng_state, restore_rng, seed_all
from .metrics import unsup_mahalanobis_scores


def effective_rank(z):
    singular = np.linalg.svd(z - z.mean(0), compute_uv=False)
    energy = singular ** 2
    if energy.sum() < 1e-20:
        return 0.0
    p = energy / energy.sum()
    return float(np.exp(-(p * np.log(np.maximum(p, 1e-20))).sum()))


@torch.no_grad()
def source_telemetry(model, files, cache, data_dir, device, teacher=None, progress_callback=None):
    saved = rng_state()
    was_training = model.training
    model.eval()
    seed_all(12345)
    rng = np.random.RandomState(12345)
    offsets = ConcentricStarTopology(star_radii=model.config.star_radii).offsets
    pad = int(np.abs(offsets).max())
    auc, ap, cnr, variance, losses, centered = [], [], [], [], [], []
    completed_batches = 0
    total_batches = len(files) * 8
    try:
        for path in files:
            array = np.load(Path(cache) / (digest(str(Path(path).resolve()))[:20] + ".npy"), mmap_mode="r")
            padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            rr, cc = rng.randint(0, 270, size=(2, 256))
            patches = padded[rr[:, None] + pad + offsets[None, :, 0],
                             cc[:, None] + pad + offsets[None, :, 1]].reshape(-1, 5, 5, 128)
            reference = torch.from_numpy(np.repeat(corner_median_ref_patch(array)[None], 32, 0)).to(device)
            zs = []
            for i in range(0, len(patches), 32):
                x = torch.from_numpy(patches[i:i + 32]).to(device)
                batch = dict(x_a=x, x_b=x, ref_a=reference, ref_b=reference,
                             scan_id=torch.zeros(len(x), dtype=torch.long, device=device))
                loss, _ = model.ssl_loss(batch, teacher)
                losses.append(float(loss))
                zs.append(model.extract_features(x).cpu().numpy())
                completed_batches += 1
                if progress_callback is not None:
                    progress_callback(completed_batches, total_batches,
                                      f"{Path(path).name} batch {i // 32 + 1}/8")
            z = np.concatenate(zs)
            centered.append(z - z.mean(0))
            spec = extract_file_metadata(path)["specimen"]
            folder = {"Corrosion": "corrosion", "Rivet_v1": "rivet_v1", "Rivet_v2": "rivet_v2"}[spec]
            gt = np.load(Path(data_dir) / "ground_truth" / folder / f"{folder}_gt_mask.npy")[rr, cc]
            score = unsup_mahalanobis_scores(z)
            valid = gt >= 0
            if len(np.unique(gt[valid])) == 2:
                auc.append(roc_auc_score(gt[valid], score[valid]))
                ap.append(average_precision_score(gt[valid], score[valid]))
                cnr.append(float((score[gt == 1].mean() - score[gt == 0].mean()) / (score[gt == 0].std() + 1e-8)))
            if (gt == 0).any():
                variance.append(float(z[gt == 0].var(0).mean()))
        return dict(val_loss=float(np.mean(losses)) if losses else None,
            val_auc=float(np.mean(auc)) if auc else None,
            val_ap=float(np.mean(ap)) if ap else None,
            val_snr=float(np.mean(cnr)) if cnr else None,
            val_sound_var=float(np.mean(variance)) if variance else None,
            effective_rank=effective_rank(np.concatenate(centered)) if centered else None,
            telemetry_role="source_validation_seen_without_labels_during_ssl")
    finally:
        model.train(was_training)
        restore_rng(saved)
