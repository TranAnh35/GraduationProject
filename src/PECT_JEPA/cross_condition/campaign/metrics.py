"""CAD is physical geometry; response maps are not segmentation ground truth."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score


def unsup_mahalanobis_scores(latents, keep_q=0.75, eps=1e-2, valid_mask=None):
    """Compute deployable anomaly scores from each scan's unlabeled background."""
    from ..downstream.test_time_adaptation import unsup_trimmed_bg_stats

    original_shape = latents.shape[:-1]
    dim = latents.shape[-1]
    flat = np.nan_to_num(np.asarray(latents).reshape(-1, dim).astype(np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)
    valid = valid_mask.reshape(-1) if valid_mask is not None else None
    mean, whitening = unsup_trimmed_bg_stats(flat, keep_q=keep_q, eps=eps, valid_mask=valid)
    distance = np.linalg.norm((flat - mean).dot(whitening), axis=-1)
    return np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0).reshape(original_shape)


def cad_objects(config):
    objects = []
    for feature in config.get("features", []):
        depth = feature.get("depth")
        kind = str(feature.get("kind", "")).lower()
        positive = float(depth) > 0 if depth is not None else (
            ("corrosion" in kind or "defect" in kind) and "only" not in kind)
        diameter = float(feature.get("diameter") or 0)
        if not positive or diameter <= 0:
            continue
        x = feature.get("corrosionX")
        y = feature.get("corrosionY")
        if x is None or y is None:
            x, y = feature.get("x"), feature.get("y")
        if x is not None and y is not None:
            objects.append({"id": str(feature["id"]), "x": float(x), "y": float(y),
                            "diameter": diameter})
    return objects


def candidates(score, threshold, stride=1, separation_px=12, max_candidates=None):
    """Uses scores only. No CAD, test labels or test-specific tolerance fitting."""
    smooth = gaussian_filter(np.asarray(score, dtype=float), sigma=0.7)
    if np.ptp(smooth) < 1e-12:
        return []
    width = max(3, int(separation_px / stride) // 2 * 2 + 1)
    keep = (smooth == maximum_filter(smooth, size=width, mode="nearest")) & (smooth >= threshold)
    rr, cc = np.where(keep)
    order = np.argsort(-smooth[rr, cc], kind="stable")[:max_candidates]
    return [{"x": float(cc[i] * stride), "y": float(rr[i] * stride),
             "score": float(smooth[rr[i], cc[i]])} for i in order]


def match_candidates(predictions, truth, tolerance_px):
    if not predictions or not truth:
        return [], len(predictions), len(truth)
    xy = np.array([[p["x"], p["y"]] for p in predictions])
    gt = np.array([[g["x"], g["y"]] for g in truth])
    distance = np.linalg.norm(xy[:, None] - gt[None], axis=-1)
    # Large penalty gives maximum-cardinality matching within tolerance first.
    cost = np.where(distance <= tolerance_px, distance, 1e6)
    ii, jj = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(distance[i, j])) for i, j in zip(ii, jj)
             if distance[i, j] <= tolerance_px]
    return pairs, len(predictions) - len(pairs), len(truth) - len(pairs)


def detection_metrics(y, score, threshold, objects, stride, tolerance_px):
    y, score = np.asarray(y), np.asarray(score)
    valid = (y == 0) | (y == 1)
    a, b = y[valid], score[valid]
    pred = b >= threshold
    tp = int(((a == 1) & pred).sum())
    fp = int(((a == 0) & pred).sum())
    fn = int(((a == 1) & ~pred).sum())
    peaks = candidates(score, threshold, stride)
    pairs, false_alarms, misses = match_candidates(peaks, objects, tolerance_px)
    return {"auc": float(roc_auc_score(a, b)) if len(np.unique(a)) == 2 else None,
            "ap": float(average_precision_score(a, b)) if (a == 1).any() else None,
            "prevalence": float((a == 1).mean()), "iou": tp / max(1, tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "fpr": fp / max(1, (a == 0).sum()),
            "cnr": float((b[a == 1].mean() - b[a == 0].mean()) / (b[a == 0].std() + 1e-8)) if (a == 1).any() and (a == 0).any() else None,
            "object_recall": len(pairs) / max(1, len(objects)),
            "false_alarms_per_scan": false_alarms, "missed_objects": misses,
            "localization_error_px": float(np.mean([p[2] for p in pairs])) if pairs else None}, peaks, pairs


def object_descriptor(features, score, x, y, stride, radius_px=20):
    """Fixed ROI independent of the unknown diameter, including the footprint.

    Coordinates supplied by a detector at inference. Source annotations may
    supply training centers. Never use target CAD diameter to choose ROI size.
    """
    rr, cc = np.indices(score.shape)
    dist = np.sqrt((rr * stride - y) ** 2 + (cc * stride - x) ** 2)
    pieces = []
    for lo, hi in [(0, radius_px / 3), (radius_px / 3, radius_px * 2 / 3),
                   (radius_px * 2 / 3, radius_px)]:
        keep = (dist >= lo) & (dist <= hi)
        if not keep.any():
            pieces.extend(np.zeros(features.shape[-1] * 2 + 3))
        else:
            z, s = features[keep], score[keep]
            pieces.extend(np.r_[z.mean(0), z.std(0), np.quantile(s, [0.1, 0.5, 0.9])])
    return np.asarray(pieces, dtype=np.float32)
