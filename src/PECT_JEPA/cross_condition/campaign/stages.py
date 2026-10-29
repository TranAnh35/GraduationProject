"""Data preparation, feature extraction, readouts and evidence aggregation."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..data.preprocessing import load_canonical_cscan_from_tdms
from ..data.split import extract_file_metadata
from ..data.topologies import ConcentricStarTopology
from ..downstream.domain_adaptation.da_bridge import ClosedFormCORAL
from ..downstream.test_time_adaptation import unsup_trimmed_bg_stats
from .common import ProgressReporter, atomic_json, digest, read_json, sha256
from .learning import load_model
from .metrics import cad_objects, detection_metrics, object_descriptor, unsup_mahalanobis_scores


def key(path):
    return digest(str(Path(path).resolve()))[:20]


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, array)
    os.replace(temp, path)


def prepare(plan, output):
    entries = plan["inventory"]["files"]
    progress = ProgressReporter(output)
    progress.update("prepare", 0, len(entries), "scan", detail="starting", force=True)
    for index, entry in enumerate(entries):
        path = entry["path"]
        cached = Path(output) / (key(path) + ".npy")
        check = cached.with_suffix(".check.json")
        progress.update("prepare", index, len(entries), "scan", detail=f"inspect {Path(path).name}", force=True)
        if cached.exists() and check.exists():
            saved = read_json(check)
            if saved["sha256"] != sha256(cached) or saved["raw_sha256"] != entry["sha256"]:
                raise ValueError(f"Corrupt canonical cache: {cached}")
            array = np.load(cached, mmap_mode="r")
            if array.shape != (270, 270, 128) or not np.isfinite(array).all():
                raise ValueError(f"Invalid canonical cache: {cached}")
            progress.update("prepare", index + 1, len(entries), "scan", detail=Path(path).name, force=True)
            continue
        # Verify raw bytes at preparation, not just path/mtime.
        progress.update("prepare", index, len(entries), "scan", detail=f"hash raw {Path(path).name}", force=True)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Raw input changed: {path}")
        progress.update("prepare", index, len(entries), "scan", detail=f"canonicalize {Path(path).name}", force=True)
        array = load_canonical_cscan_from_tdms(path, temporal_samples=128,
            transform_csv=plan["settings"]["transform_csv"], allow_identity_fallback=False)
        if array.shape != (270, 270, 128) or not np.isfinite(array).all():
            raise ValueError(f"Invalid C-scan: {path}")
        save_array(cached, array)
        atomic_json(check, {"sha256": sha256(cached), "raw_sha256": entry["sha256"]})
        print(f"prepared {Path(path).name}", flush=True)
        progress.update("prepare", index + 1, len(entries), "scan", detail=Path(path).name, force=True)
    progress.update("prepare", len(entries), len(entries), "scan", detail="complete", force=True)
    atomic_json(Path(output) / "result.json", {"files": len(plan["inventory"]["files"]),
        "normalization": "global_peak", "shape": [270, 270, 128]})


def patch_batches(array, stride, batch_size, radii=(2, 4, 7)):
    offsets = ConcentricStarTopology(star_radii=tuple(radii)).offsets
    pad = int(np.abs(offsets).max())
    padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    rr, cc = np.meshgrid(np.arange(0, array.shape[0], stride),
                         np.arange(0, array.shape[1], stride), indexing="ij")
    r, c = rr.ravel() + pad, cc.ravel() + pad
    for start in range(0, len(r), batch_size):
        patches = padded[r[start:start + batch_size, None] + offsets[None, :, 0],
                         c[start:start + batch_size, None] + offsets[None, :, 1]]
        yield start, patches.reshape(-1, 5, 5, 128)


def extract(job, plan, output, device):
    protocol, settings = plan["protocols"][job["scenario"]], plan["settings"]
    signal_features = job["representation"] in ("spatial_fft", "dualview")
    model = load_model(job["checkpoint"], device).eval() if not signal_features else None
    stride = settings["feature_stride"]
    h = len(range(0, 270, stride))
    paths = protocol["source"] + protocol["target"]
    total_work = len(paths) * h * h
    completed_work = 0
    progress = ProgressReporter(output)
    progress.update("extract", 0, total_work, "patch", detail="starting", force=True)
    for file_index, path in enumerate(paths):
        progress.update("extract", completed_work, total_work, "patch",
                        detail=f"scan {file_index + 1}/{len(paths)}: {Path(path).name}", force=True)
        cached = Path(output) / (key(path) + ".npy")
        check = cached.with_suffix(".check.json")
        if cached.exists() and check.exists():
            try:
                valid = read_json(check)["sha256"] == sha256(cached)
                if valid:
                    z = np.load(cached, mmap_mode="r")
                    valid = z.ndim == 3 and z.shape[:2] == (h, h) and np.isfinite(z).all()
            except (KeyError, OSError, TypeError, ValueError):
                valid = False
            if valid:
                completed_work = (file_index + 1) * h * h
                progress.update("extract", completed_work, total_work, "patch",
                                detail=f"cached {Path(path).name}", force=True)
                continue
            # An interrupted write or damaged disposable cache is regenerated
            # from the locked checkpoint instead of blocking campaign resume.
            cached.unlink(missing_ok=True)
            check.unlink(missing_ok=True)
        array = np.load(Path(plan["prepare_dir"]) / (key(path) + ".npy"), mmap_mode="r")
        features = []
        with torch.inference_mode():
            for _, patches in patch_batches(array, stride, settings["extract_batch_size"]):
                if model is None:
                    flat = patches.reshape(-1, 25, 128)
                    spectral = np.log1p(np.abs(np.fft.rfft(flat, norm="ortho")))
                    # Same 25 spatial probes as JEPA; no single-pixel strawman.
                    if job["representation"] == "dualview":
                        ft = np.fft.rfft(flat, norm="ortho")
                        unit = ft / np.maximum(np.abs(ft), 1e-8)
                        frequency = np.concatenate([spectral, unit.real, unit.imag], -1)
                        z = np.concatenate([flat[:, 0], flat[:, 1:].mean(1),
                            frequency[:, 0], frequency[:, 1:].mean(1)], axis=1)
                    else:
                        z = np.concatenate([spectral[:, 0], spectral[:, 1:].mean(1)], axis=1)
                else:
                    z = model.extract_features(torch.from_numpy(patches).to(device)).cpu().numpy()
                features.append(z.astype(np.float32))
                completed_work += len(patches)
                progress.update("extract", completed_work, total_work, "patch",
                    detail=f"scan {file_index + 1}/{len(paths)}: {Path(path).name}")
        dtype = job.get("feature_dtype", settings.get("feature_dtype", "float32"))
        save_array(cached, np.concatenate(features).reshape(h, h, -1).astype(dtype))
        atomic_json(check, {"sha256": sha256(cached)})
        print(f"extracted {Path(path).name}", flush=True)
        progress.update("extract", (file_index + 1) * h * h, total_work, "patch",
                        detail=f"saved {Path(path).name}", force=True)
        completed_work = (file_index + 1) * h * h
    progress.update("extract", total_work, total_work, "patch", detail="complete", force=True)
    atomic_json(Path(output) / "result.json", {"files": len(paths),
        "feature_stride": stride, "representation": job["representation"],
        "feature_storage_dtype": job.get("feature_dtype", settings.get("feature_dtype", "float32")),
        "pixel_metrics_scope": "sampled_canonical_grid" if stride > 1 else "dense_canonical_grid"})


def ground_truth(plan):
    root = Path(plan["settings"]["data_dir"]) / "ground_truth"
    return {specimen: {"mask": np.load(root / folder / f"{folder}_gt_mask.npy"),
            "objects": cad_objects(read_json(root / folder / f"{folder}_flaws_config.json"))}
            for specimen, folder in [("Corrosion", "corrosion"), ("Rivet_v1", "rivet_v1"),
                                     ("Rivet_v2", "rivet_v2")]}


def feature(path, directory):
    return np.load(Path(directory) / (key(path) + ".npy"), mmap_mode="r").astype(np.float32, copy=False)


def whiten(z):
    shape = z.shape
    flat = np.asarray(z).reshape(-1, shape[-1])
    mu, w = unsup_trimmed_bg_stats(flat, keep_q=0.75, eps=0.01)
    return ((flat - mu) @ w).reshape(shape).astype(np.float32)


def sample_pool(paths, directory, gt, stride, seed, cap, labels=True, transform=None, spatial=None,
                progress_callback=None):
    xs, ys = [], []
    for i, path in enumerate(paths):
        z = feature(path, directory)
        if transform:
            z = transform(z)
        flat = z.reshape(-1, z.shape[-1])
        mask = gt[extract_file_metadata(path)["specimen"]]["mask"][::stride, ::stride].ravel() if labels else None
        idx = np.flatnonzero(mask >= 0) if labels else np.arange(len(flat))
        if spatial is not None:
            idx = idx[spatial[::stride, ::stride].ravel()[idx]]
        rng = np.random.RandomState(seed + i)
        idx = rng.choice(idx, min(cap, len(idx)), replace=False)
        xs.append(flat[idx])
        if labels:
            ys.append(mask[idx])
        if progress_callback is not None:
            progress_callback(i + 1, len(paths), Path(path).name)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(int) if labels else None


def evaluate(job, plan, output):
    """Per-file resume; thresholds fit on source validation only."""
    output = Path(output)
    progress = ProgressReporter(output)
    progress.update("load_ground_truth", 0, 1, "step", detail="load CAD masks", force=True)
    protocol, cfg = plan["protocols"][job["scenario"]], plan["settings"]
    stride, arm, directory = cfg["feature_stride"], job["arm"], job["features"]
    fit_directory = job.get("fit_features", directory)
    gt = ground_truth(plan)
    progress.update("load_ground_truth", 1, 1, "step", detail="complete", force=True)
    spatial_train = spatial_test = None
    if protocol.get("spatial_holdout"):
        from .protocol import spatial_masks
        spatial_train, spatial_test = spatial_masks((270, 270))
    transformed = whiten if arm == "whiten_lr" else None
    progress.update("sample_source_fit", 0, len(protocol["fit"]), "scan", detail="sampling source labels", force=True)
    x, y = sample_pool(protocol["fit"], fit_directory, gt, stride, job["seed"],
        cfg["probe_samples_per_file"], transform=transformed, spatial=spatial_train,
        progress_callback=lambda current, total, detail: progress.update(
            "sample_source_fit", current, total, "scan", detail=detail, force=True))
    coral = None
    if arm == "coral":
        # Domain marginals retain natural prevalence; no target GT filtering.
        progress.update("sample_coral_source", 0, len(protocol["fit"]), "scan", force=True)
        xs, _ = sample_pool(protocol["fit"], directory, gt, stride, job["seed"], 1000,
            labels=False, spatial=spatial_train, progress_callback=lambda current, total, detail:
                progress.update("sample_coral_source", current, total, "scan", detail=detail, force=True))
        progress.update("sample_coral_target", 0, len(protocol["target"]), "scan", force=True)
        xt, _ = sample_pool(protocol["target"], directory, gt, stride, job["seed"], 1000,
            labels=False, spatial=spatial_train, progress_callback=lambda current, total, detail:
                progress.update("sample_coral_target", current, total, "scan", detail=detail, force=True))
        progress.update("fit_coral", 0, 1, "step", detail="align source and target covariance", force=True)
        coral = ClosedFormCORAL(ridge=0.1).fit(xs, xt)
        x = coral.transform_source(x)
        progress.update("fit_coral", 1, 1, "step", detail="complete", force=True)
    progress.update("fit_logistic_readout", 0, 1, "step", detail="fit source classifier", force=True)
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=job["seed"])
    lr.fit(scaler.transform(x), y)
    progress.update("fit_logistic_readout", 1, 1, "step", detail="complete", force=True)
    neural = None
    if arm in ("neural_control", "dann", "tf_source", "tf_consistency_pect"):
        from .adaptation import fit_readout
        progress.update("sample_neural_source", 0, len(protocol["fit"]), "scan", force=True)
        xs, _ = sample_pool(protocol["fit"], fit_directory, gt, stride, job["seed"], 1000,
            labels=False, spatial=spatial_train, progress_callback=lambda current, total, detail:
                progress.update("sample_neural_source", current, total, "scan", detail=detail, force=True))
        progress.update("sample_neural_target", 0, len(protocol["target"]), "scan", force=True)
        xt, _ = sample_pool(protocol["target"], directory, gt, stride, job["seed"], 1000,
            labels=False, spatial=spatial_train, progress_callback=lambda current, total, detail:
                progress.update("sample_neural_target", current, total, "scan", detail=detail, force=True))
        neural_total = cfg["neural_source_steps"] if arm == "tf_source" else (
            cfg["neural_source_steps"] + cfg["neural_adapt_steps"])
        progress.update("neural_readout", 0, neural_total, "step", detail=arm, force=True)
        neural = fit_readout(output, scaler.transform(x).astype(np.float32), y,
            scaler.transform(xs).astype(np.float32), scaler.transform(xt).astype(np.float32),
            job["seed"], arm, cfg)

    def score_file(path, is_source=False):
        z = feature(path, fit_directory if is_source else directory)
        if arm == "maha":
            return unsup_mahalanobis_scores(z, keep_q=0.75, eps=0.01).reshape(z.shape[:2])
        zz = transformed(z) if transformed else z
        flat = zz.reshape(-1, zz.shape[-1])
        if coral is not None and is_source:
            flat = coral.transform_source(flat)
        if neural is not None:
            from .adaptation import predict
            return predict(neural, scaler.transform(flat).astype(np.float32)).reshape(z.shape[:2])
        return lr.predict_proba(scaler.transform(flat))[:, 1].reshape(z.shape[:2])

    negatives = []
    progress.update("source_validation_threshold", 0, len(protocol["val"]), "scan", force=True)
    for index, path in enumerate(protocol["val"]):
        mask = gt[extract_file_metadata(path)["specimen"]]["mask"][::stride, ::stride]
        if spatial_train is not None:
            mask = np.where(spatial_train[::stride, ::stride], mask, -1)
        scores = score_file(path, True)
        negatives.append(scores[mask == 0])
        progress.update("source_validation_threshold", index + 1, len(protocol["val"]),
                        "scan", detail=Path(path).name, force=True)
    threshold = float(np.quantile(np.concatenate(negatives), 1 - cfg["validation_fpr"]))
    atomic_json(output / "threshold.json", {"threshold": threshold,
        "fit_role": "source_validation_background", "requested_pixel_fpr": cfg["validation_fpr"],
        "target_labels_used": False, "transfer_note": "source threshold; target FPR is measured, not guaranteed"})

    # Object-level sizing. Train only on source CAD centers; fixed-size response
    # descriptors include footprint and background. Target CAD only scores.
    desc, diameters = [], []
    sizing_paths = [path for path in protocol["fit"]
                    if extract_file_metadata(path)["specimen"] == "Corrosion"]
    progress.update("fit_sizing", 0, len(sizing_paths), "scan", force=True)
    for index, path in enumerate(sizing_paths):
        spec = extract_file_metadata(path)["specimen"]
        z, score = feature(path, fit_directory), score_file(path, True)
        for obj in gt[spec]["objects"]:
            if spatial_train is not None and not spatial_train[int(obj["y"]), int(obj["x"])]:
                continue
            desc.append(object_descriptor(z, score, obj["x"], obj["y"], stride, job.get("sizing_radius_px", 20)))
            diameters.append(obj["diameter"])
        progress.update("fit_sizing", index + 1, len(sizing_paths), "scan", detail=Path(path).name, force=True)
    size_scaler = StandardScaler().fit(desc)
    sizing = Ridge(alpha=10).fit(size_scaler.transform(desc), diameters)
    lower, upper = float(min(diameters)), float(max(diameters))
    constant = float(np.median(diameters))
    rows, objects, known_objects = [], [], []
    progress.update("score_target", 0, len(protocol["target"]), "scan", detail="starting", force=True)
    for target_index, path in enumerate(protocol["target"]):
        progress.update("score_target", target_index, len(protocol["target"]), "scan",
                        detail=Path(path).name, force=True)
        record = output / "scans" / (key(path) + ".json")
        record_check = record.with_suffix(".check.json")
        if record.exists() and record_check.exists():
            if read_json(record_check)["sha256"] != sha256(record):
                raise ValueError(f"Corrupt partial scan result: {record}")
            saved = read_json(record)
            rows.append(saved["metrics"])
            objects.extend(saved["objects"])
            known_objects.extend(saved.get("known_location_objects", []))
            progress.update("score_target", target_index + 1, len(protocol["target"]), "scan",
                            detail=f"cached {Path(path).name}", force=True)
            continue
        spec = extract_file_metadata(path)["specimen"]
        z, score = feature(path, directory), score_file(path)
        labels = gt[spec]["mask"][::stride, ::stride]
        truth_objects = gt[spec]["objects"]
        detect_score = score
        if spatial_test is not None:
            labels = np.where(spatial_test[::stride, ::stride], labels, -1)
            truth_objects = [g for g in truth_objects if spatial_test[int(g["y"]), int(g["x"])]]
            detect_score = np.where(spatial_test[::stride, ::stride], score, -1e6)
        metrics, peaks, pairs = detection_metrics(labels, detect_score,
            threshold, truth_objects, stride, cfg["localization_tolerance_px"])
        metrics.update(file=path, scenario=job["scenario"], seed=job["seed"], arm=arm,
                       representation=job["representation"], **{k: v for k, v in extract_file_metadata(path).items() if k != "file_path"})
        details = []
        known = []
        if spec == "Corrosion":
            for truth in truth_objects:
                d = object_descriptor(z, score, truth["x"], truth["y"], stride, job.get("sizing_radius_px", 20))
                raw = float(sizing.predict(size_scaler.transform(d[None]))[0])
                known.append({"file": path, "object_id": truth["id"],
                    "scope": "known_location_diagnostic_not_deployable_detection",
                    "true_diameter_mm": truth["diameter"], "pred_diameter_mm": float(np.clip(raw, lower, upper)),
                    "unclipped_diameter_mm": raw, "absolute_error_mm": abs(float(np.clip(raw, lower, upper)) - truth["diameter"])})
            metrics["sizing_known_location_mae_mm"] = float(np.mean([d["absolute_error_mm"] for d in known])) if known else None
            for pi, gi, distance in pairs:
                pred, truth = peaks[pi], truth_objects[gi]
                d = object_descriptor(z, score, pred["x"], pred["y"], stride, job.get("sizing_radius_px", 20))
                raw = float(sizing.predict(size_scaler.transform(d[None]))[0])
                diameter = float(np.clip(raw, lower, upper))
                details.append({"file": path, "object_id": truth["id"],
                    "true_diameter_mm": truth["diameter"], "pred_diameter_mm": diameter,
                    "unclipped_diameter_mm": raw, "absolute_error_mm": abs(diameter - truth["diameter"]),
                    "constant_error_mm": abs(constant - truth["diameter"]),
                    "localization_error_px": distance})
            metrics["sizing_detected_mae_mm"] = float(np.mean([d["absolute_error_mm"] for d in details])) if details else None
            metrics["constant_detected_mae_mm"] = float(np.mean([d["constant_error_mm"] for d in details])) if details else None
            yt = np.asarray([d["true_diameter_mm"] for d in details])
            yp = np.asarray([d["pred_diameter_mm"] for d in details])
            metrics["sizing_detected_r2"] = float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()) if len(yt) > 1 and yt.var() > 0 else None
            metrics["sizing_detected_count"] = len(details)
            metrics["sizing_total_count"] = len(truth_objects)
        atomic_json(record, {"metrics": metrics, "objects": details, "known_location_objects": known})
        atomic_json(record_check, {"sha256": sha256(record)})
        rows.append(metrics)
        objects.extend(details)
        known_objects.extend(known)
        print(f"scored {Path(path).name}", flush=True)
        progress.update("score_target", target_index + 1, len(protocol["target"]), "scan",
                        detail=Path(path).name, force=True)
    write_csv(output / "master_benchmark_metrics.csv", rows)
    write_csv(output / "object_sizing.csv", objects)
    write_csv(output / "sizing_known_location.csv", known_objects)
    summary = {"scans": len(rows), "sizing_objects": len(objects),
        "sizing_scope": "Corrosion, detected candidates, fixed response ROI; misses separately reported",
        "sizing_prediction_bounds_mm": [lower, upper], "threshold": threshold,
        "arm": arm, "macro_ap": float(np.mean([r["ap"] for r in rows if r["ap"] is not None])),
        "macro_auc": float(np.mean([r["auc"] for r in rows if r["auc"] is not None]))}
    atomic_json(output / "master_benchmark_summary.json", summary)
    atomic_json(output / "result.json", {"complete": True, "scans": len(rows)})
    progress.update("evaluation_complete", 1, 1, "stage", detail=arm, force=True)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)


def field_mean(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
    if any(not np.isfinite(v) for v in values):
        raise ValueError(f"Nonfinite value in metric {field}")
    return float(np.mean(values)) if values else None


def aggregate(plan, root, state, verbose=True):
    rows, missing, sizing = [], [], []
    eval_jobs = [job for job in plan["jobs"] if job["kind"] == "evaluate"]
    for job in tqdm(eval_jobs, desc="Aggregate evaluation jobs", unit="job",
                    dynamic_ncols=True, disable=not verbose or not eval_jobs):
        output = Path(root) / "jobs" / job["id"] / "master_benchmark_metrics.csv"
        if state["jobs"].get(job["id"], {}).get("status") == "completed" and output.exists():
            rows.extend(csv.DictReader(output.open(encoding="utf-8")))
            objects_path = output.parent / "object_sizing.csv"
            if objects_path.exists():
                for row in csv.DictReader(objects_path.open(encoding="utf-8")):
                    sizing.append({**row, **{k: job[k] for k in ["scenario", "representation", "arm", "seed"]}})
        else:
            missing.append(job["id"])
    write_csv(Path(root) / "all_scan_metrics.csv", rows)
    write_csv(Path(root) / "all_object_sizing.csv", sizing)
    grouped = {}
    for row in rows:
        group = (row["scenario"], row["representation"], row["arm"])
        grouped.setdefault(group, []).append(row)
    summary = []
    for group, values in sorted(grouped.items()):
        summary.append(dict(zip(["scenario", "representation", "arm"], group),
            scan_seed_rows=len(values), unique_scans=len({v["file"] for v in values}),
            **{f"mean_{metric}": field_mean(values, metric)
               for metric in ["ap", "auc", "cnr", "object_recall", "false_alarms_per_scan", "iou",
                              "sizing_detected_mae_mm", "sizing_known_location_mae_mm", "sizing_detected_count", "sizing_total_count"]}))
    write_csv(Path(root) / "summary.csv", summary)
    slices = {}
    for row in rows:
        for axis in ["sensor", "waveform", "liftoff", "specimen"]:
            group = (row["scenario"], row["representation"], row["arm"], axis, row[axis])
            slices.setdefault(group, []).append(row)
    slice_rows = []
    for group, values in sorted(slices.items()):
        slice_rows.append(dict(zip(["scenario", "representation", "arm", "axis", "level"], group),
            scan_seed_rows=len(values), unique_scans=len({v["file"] for v in values}),
            **{f"mean_{metric}": field_mean(values, metric)
               for metric in ["ap", "auc", "object_recall", "false_alarms_per_scan"]}))
    write_csv(Path(root) / "by_condition.csv", slice_rows)
    # Paired adaptation effects: first average seeds for each scan, never count
    # 3 repeated seeds as 3 independent physical specimens.
    index = {(r["scenario"], r["representation"], r["arm"], r["seed"], r["file"]): r for r in rows}
    differences = []
    for row in rows:
        reference_arm = {"coral": "raw_lr", "whiten_lr": "raw_lr", "maha": "raw_lr",
                         "dann": "neural_control", "tf_consistency_pect": "tf_source"}.get(row["arm"])
        reference_rep = row["representation"]
        comparison = "readout_change" if row["arm"] == "maha" else "adaptation"
        if row["representation"] == "paired_jepa_adapt" and row["arm"] == "raw_lr":
            reference_rep, reference_arm = "paired_jepa_source", "raw_lr"
        if row["representation"] == "paired_jepa_source_float32":
            reference_rep, reference_arm, comparison = "paired_jepa_source", row["arm"], "storage_precision"
        if row["representation"] in ("self_jepa_source", "spatial_mae_source", "random_source",
            "spatial_fft", "paired_jepa_transductive", "paired_jepa_no_condition") and row["arm"] == "raw_lr":
            reference_rep, reference_arm, comparison = "paired_jepa_source", "raw_lr", "representation_or_pretrain_scope"
        if not reference_arm:
            continue
        base = index.get((row["scenario"], reference_rep, reference_arm, row["seed"], row["file"]))
        if base:
            differences.append({"scenario": row["scenario"], "representation": row["representation"],
                "arm": row["arm"], "seed": row["seed"], "file": row["file"],
                "reference_arm": reference_arm, "reference_representation": reference_rep,
                "comparison": comparison,
                "delta_ap": float(row["ap"]) - float(base["ap"]),
                "delta_auc": float(row["auc"]) - float(base["auc"])})
    write_csv(Path(root) / "paired_deltas.csv", differences)
    negative = {}
    for row in differences:
        group = (row["scenario"], row["representation"], row["arm"], row["comparison"])
        negative.setdefault(group, {}).setdefault(row["file"], []).append(row["delta_ap"])
    negative_rows = []
    for group, scans in sorted(negative.items()):
        values = np.array([np.mean(v) for v in scans.values()])
        negative_rows.append(dict(zip(["scenario", "representation", "arm", "comparison"], group),
            seed_averaged_scans=len(values), mean_delta_ap=float(values.mean()),
            negative_transfer_rate=float((values < 0).mean()), worst_scan_delta_ap=float(values.min())))
    write_csv(Path(root) / "negative_transfer.csv", negative_rows)
    atomic_json(Path(root) / "completion.json", {"complete": not missing,
        "missing_jobs": missing, "scan_seed_rows": len(rows),
        "uncertainty": "scan summaries are descriptive repeated measurements of 3 specimen geometries; not population CIs"})
    status = "COMPLETE" if not missing else "INCOMPLETE"
    (Path(root) / "REPORT.md").write_text(
        f"# PECT campaign: {status}\n\n"
        f"Completed evaluation jobs: {sum(j['kind'] == 'evaluate' for j in plan['jobs']) - len(missing)}\n\n"
        f"Missing evaluation jobs: {len(missing)}. See completion.json and state.json.\n\n"
        "Sources: jobs/*/master_benchmark_metrics.csv and master_benchmark_summary.json.\n\n"
        "Tables: summary.csv, by_condition.csv, paired_deltas.csv, negative_transfer.csv.\n\n"
        "Sizing: jobs/*/object_sizing.csv contains matched detections only; report missed_objects and sizing_total_count alongside MAE.\n\n"
        "Scope: existing physical specimens. Seed repeats and scan repeats do not establish independent-specimen generalization.\n\n"
        "tf_consistency_pect is a PECT adaptation inspired by TFDA; it is not the official TFDA implementation.\n\n"
        "Completion means artifacts are available, not that a scientific hypothesis was accepted.\n",
        encoding="utf-8")
