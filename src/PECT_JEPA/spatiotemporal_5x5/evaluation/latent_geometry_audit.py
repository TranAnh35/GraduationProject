"""Latent Geometry Audit for PECT-JEPA (frozen EXP-28 checkpoint).

Implements Experiments A-F on the frozen 5x5 PECT-JEPA encoder. No architecture
change, no encoder training. Everything downstream (linear probes, CORAL,
canonicalization) is applied on frozen representation output.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import (
    get_ground_truth_manager,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata
from src.PECT_JEPA.spatiotemporal_5x5.evaluate import load_model_from_checkpoint
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.anomaly_detection import (
    roc_auc_score as _sklearn_roc_auc_score,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import (
    apply_lowpass_filter,
    linear_time_resample,
    moving_rms_envelope,
    normalize_waveforms_linear,
    read_tdms_1d_waveforms,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUDIT_DIR_DEFAULT = "experiments/5x5/exp28_full_20ep/latent_geometry_audit"
WAVEFORMS = ["Square", "Gaussian", "Chirp"]


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """NaN-safe wrapper around sklearn roc_auc_score."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(_sklearn_roc_auc_score(y_true, y_score))


class FileManifest:
    """Discover and index every TDMS file with its scientific metadata."""

    def __init__(
        self,
        data_root: str = "data",
        specimen: Optional[str] = None,
        sensor: Optional[str] = None,
        waveform: Optional[str] = None,
        liftoff: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.specimen = specimen
        self.sensor = sensor
        self.waveform = waveform
        self.liftoff = liftoff
        self.max_files = max_files
        self.files: List[Path] = []
        self._meta_by_path: Dict[str, Dict[str, Any]] = {}
        self._build()

    def _build(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        items: List[Tuple[Path, Dict[str, Any]]] = []
        counts: Dict[str, int] = collections.Counter()
        for tdms in sorted(self.data_root.rglob("*.tdms")):
            if tdms.suffix == ".tdms_index":
                continue
            meta = extract_file_metadata(str(tdms))
            if self.specimen and meta["specimen"] != self.specimen:
                continue
            if self.sensor and meta["sensor"] != self.sensor:
                continue
            if self.waveform and meta["waveform"] != self.waveform:
                continue
            if self.liftoff and meta["liftoff"] != self.liftoff:
                continue
            items.append((tdms, meta))
            counts[meta["waveform"]] += 1

        items.sort(key=lambda kv: (
            kv[1]["sensor"], kv[1]["specimen"], kv[1]["waveform"], kv[0].name,
        ))

        self.files = [p for p, _ in items]
        self._meta_by_path = {str(p): m for p, m in items}

        if self.max_files is not None and len(self.files) > self.max_files:
            self.files = self.files[: self.max_files]
            self._meta_by_path = {
                str(p): m for p, m in self._meta_by_path.items() if p in set(self.files)
            }

        self._counts = counts
        if self.files:
            self._counts = collections.Counter(
                m["waveform"] for m in self._meta_by_path.values()
            )

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterable[Path]:
        return iter(self.files)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_files": len(self.files),
            "waveform_counts": self._counts,
            "file_paths": [str(p) for p in self.files],
        }
# --------------------------------------------------------------------------- #
# Patch sampling (defect / sound centres on the standardized [270, 270] C-scan)
# --------------------------------------------------------------------------- #
class PatchSampler:
    """Samples defect + sound patch centres for every file in a manifest.

    Defect centres come from the CAD ground-truth (cropped coordinates); sound
    centres are uniform samples of mask==0 pixels that leave a valid
    [5, 5, C] patch inside the standardized grid.
    """

    def __init__(self, gt_mgr: Any, sound_per_file: int = 8, seed: int = 42) -> None:
        self.gt_mgr = gt_mgr
        self.sound_per_file = sound_per_file
        self.rng = np.random.default_rng(seed)

    def sample_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Return a list of patch requests for one TDMS file."""
        meta = extract_file_metadata(file_path)
        specimen = meta["specimen"]
        grid = self.gt_mgr.get_ground_truth_mask_for_file(file_path, aligned_scan=True)  # [270,270]
        h, w = grid.shape
        requests: List[Dict[str, Any]] = []

        # 1. Defect centres from CAD features (already in cropped coordinates)
        try:
            flaw_feats = self.gt_mgr.get_flaw_features(specimen, coordinate_system="cropped")
        except Exception:
            flaw_feats = []
        for feat in flaw_feats:
            cx = feat.get("corrosionX", feat.get("x", np.nan))
            cy = feat.get("corrosionY", feat.get("y", np.nan))
            if cx is None or cy is None or np.isnan(cx) or np.isnan(cy):
                continue
            cx = int(np.clip(int(round(float(cx))), 2, h - 3))
            cy = int(np.clip(int(round(float(cy))), 2, w - 3))
            requests.append({"row": cy, "col": cx, "kind": "defect"})

        # 2. Uniform sound centres (avoid boundary and defect bands)
        sound_candidates = np.argwhere(grid == 0)
        if len(sound_candidates) == 0:
            sound_candidates = np.argwhere(grid >= 0)
        if len(sound_candidates) > 0:
            border = 2
            valid = sound_candidates[
                (sound_candidates[:, 0] >= border)
                & (sound_candidates[:, 0] < h - border)
                & (sound_candidates[:, 1] >= border)
                & (sound_candidates[:, 1] < w - border)
            ]
            if len(valid) > 0:
                n_sound = min(self.sound_per_file, len(valid))
                idxs = self.rng.choice(len(valid), size=n_sound, replace=False)
                for i in idxs:
                    r, c = valid[i]
                    requests.append({"row": int(r), "col": int(c), "kind": "sound"})

        return requests

    def sample_all(self, files: Sequence[str]) -> List[Dict[str, Any]]:
        """Sample patch requests for a list of file paths (indexed in one list)."""
        out: List[Dict[str, Any]] = []
        for fp in files:
            for req in self.sample_for_file(str(fp)):
                req = dict(req)
                req["file_path"] = str(fp)
                out.append(req)
        return out


# --------------------------------------------------------------------------- #
# Batched frozen-encoder feature extraction
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Extracts frozen PECT-JEPA representations at a set of sampled patches.

    Outputs
    -------
    pooled   : np.ndarray  [N, 128]  foundation representation (2 x D)
    pooled_std : np.ndarray  [N, 128] foundation representation, self-calibrated
    tile     : np.ndarray  [N, 25, 64] per-tile context-encoder latents (Exp B)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 512,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.patch_h = 5
        self.patch_w = 5
        self.out_dim = 2 * self.model.config.embed_dim  # 128 for exp28 (D=64)

    def _patch_grid(self, grid: np.ndarray, row: int, col: int) -> np.ndarray:
        """[5, 5, C] patch centred at (row, col) of a [H, W, C] grid."""
        r0 = row - self.patch_h // 2
        c0 = col - self.patch_w // 2
        return np.asarray(
            grid[r0 : r0 + self.patch_h, c0 : c0 + self.patch_w], dtype=np.float32
        )

    @torch.no_grad()
    def extract_batch(
        self, patches: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a batch of patches through the frozen encoder (3 forward passes)."""
        x = torch.from_numpy(patches).float().to(self.device)
        with torch.inference_mode():
            z_raw = (
                self.model.extract_foundation_representation(x, self_calibrate=False)
                .cpu()
                .numpy()
            )
            z_std = (
                self.model.extract_foundation_representation(x, self_calibrate=True)
                .cpu()
                .numpy()
            )
            z_tile = (
                self.model.extract_all_features(x)
                .cpu()
                .numpy()
            )
        return z_raw, z_std, z_tile

    @torch.no_grad()
    def extract_requests(
        self,
        requests: Sequence[Dict[str, Any]],
        grid_loader: Callable[[str], np.ndarray],
        progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Extract features at the given patch requests (batched over files)."""
        by_file: Dict[str, List[Tuple[int, Dict[str, Any]]]] = collections.defaultdict(list)
        for i, req in enumerate(requests):
            by_file[req["file_path"]].append((i, req))

        grids_cache: Dict[str, np.ndarray] = {}
        for fp, _ in by_file.items():
            grids_cache[str(fp)] = grid_loader(str(fp))

        order: List[int] = []
        for i, req in enumerate(requests):
            row, col = int(req["row"]), int(req["col"])
            if row - self.patch_h // 2 < 0 or col - self.patch_w // 2 < 0:
                continue
            order.append(i)

        n = len(order)
        pooled = np.zeros((n, self.out_dim), dtype=np.float32)
        pooled_std = np.zeros((n, self.out_dim), dtype=np.float32)
        tile = np.zeros((n, self.patch_h * self.patch_w, self.model.config.embed_dim), dtype=np.float32)

        if n == 0:
            return {
                "file_id": np.zeros(0, dtype=np.int64),
                "file_path": np.array([], dtype=object),
                "row": np.zeros(0, dtype=np.int64),
                "col": np.zeros(0, dtype=np.int64),
                "kind": np.array([], dtype=object),
                "pooled": pooled,
                "pooled_std": pooled_std,
                "tile": tile,
            }

        all_patches: List[np.ndarray] = []
        meta: List[Tuple[int, str, str]] = []
        for i in order:
            req = requests[i]
            grid = grids_cache[str(req["file_path"])]
            patch = self._patch_grid(grid, req["row"], req["col"])[np.newaxis, ...]
            all_patches.append(patch)
            meta.append((i, req.get("kind", "unknown"), str(req["file_path"])))

        patches = np.concatenate(all_patches, axis=0)
        z_raw, z_std, z_tile = self.extract_batch(patches)

        for k, (i, kind, fp) in enumerate(meta):
            pooled[i, :] = z_raw[k]
            pooled_std[i, :] = z_std[k]
            tile[i, :, :] = z_tile[k]

        return {
            "file_id": np.array([meta[k][0] for k in range(len(meta))], dtype=np.int64),
            "file_path": np.array([meta[k][2] for k in range(len(meta))], dtype=object),
            "row": np.array([requests[meta[k][0]]["row"] for k in range(len(meta))], dtype=np.int64),
            "col": np.array([requests[meta[k][0]]["col"] for k in range(len(meta))], dtype=np.int64),
            "kind": np.array([meta[k][1] for k in range(len(meta))], dtype=object),
            "pooled": pooled,
            "pooled_std": pooled_std,
            "tile": tile,
        }


# --------------------------------------------------------------------------- #
# Latent normalizations + linear probe utilities (train stats only)
# --------------------------------------------------------------------------- #
class NormalizationTransformer:
    """Applies Raw / Center / Standardize / Whiten to a pooled feature matrix.

    All statistics are computed on the supplied TRAIN matrix (avoids leakage)
    and applied identically to train and test.
    """

    NAME_RAW = "raw"
    NAME_CENTER = "centered"
    NAME_STD = "standardized"
    NAME_WHITEN = "whitened"

    def __init__(self, train_features: np.ndarray) -> None:
        self.mu = train_features.mean(axis=0)
        self.sigma = train_features.std(axis=0) + 1e-8
        cov = np.cov(train_features, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        self.whiten_mat = (eigvecs / np.sqrt(eigvals)[None, :]) @ eigvecs.T

    def transform(self, features: np.ndarray, version: str) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if version == self.NAME_RAW:
            return x
        if version == self.NAME_CENTER:
            return x - self.mu[None, :]
        if version == self.NAME_STD:
            return (x - self.mu[None, :]) / self.sigma[None, :]
        if version == self.NAME_WHITEN:
            return (x - self.mu[None, :]) @ self.whiten_mat.T
        raise ValueError(f"Unknown normalization version: {version}")

    @staticmethod
    def coral(src: np.ndarray, tgt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Standard CORAL: align source covariance/mean to target distribution."""
        m = src.shape[1]
        cov_s = np.cov(src, rowvar=False) + eps * np.eye(m)
        cov_t = np.cov(tgt, rowvar=False) + eps * np.eye(m)
        mu_s, mu_t = src.mean(axis=0), tgt.mean(axis=0)
        eigvals_t, eigvecs_t = np.linalg.eigh(cov_t)
        eigvals_t = np.maximum(eigvals_t, 1e-6)
        whiten_t = (eigvecs_t / np.sqrt(eigvals_t)[None, :]) @ eigvecs_t.T
        src_white = (src - mu_s[None, :]) @ whiten_t.T
        cov_t_sqrt_inv = (eigvecs_t * np.sqrt(eigvals_t)[None, :]) @ eigvecs_t.T
        return src_white @ cov_t_sqrt_inv.T + mu_t[None, :]


def fit_linear_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 42) -> Any:
    """Binary logistic regression probe."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, solver="liblinear", max_iter=4000, random_state=seed, class_weight="balanced")
    clf.fit(X, y)
    return clf


def fit_mlp_probe(X: np.ndarray, y: np.ndarray, hidden: Tuple[int, ...] = (64,), seed: int = 42) -> Any:
    """Small MLP probe (sklearn), e.g. D->64->1 or D->32->32->1."""
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=3000,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=60,
    )
    clf.fit(X, y)
    return clf


class LatentGeometryAudit:
    """Runs Experiments A-F on the frozen EXP-28 checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        data_root: str = "data",
        device: str = "cuda",
        specimen: Optional[str] = None,
        sensor: Optional[str] = None,
        sound_per_file: int = 8,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.device = device
        self.specimen = specimen
        self.sensor = sensor
        self.sound_per_file = sound_per_file
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.manifest = FileManifest(
            data_root=data_root,
            specimen=specimen,
            sensor=sensor,
            waveform=None,
        )
        self.gt_mgr = get_ground_truth_manager(data_dir=data_root)
        self.sampler = PatchSampler(self.gt_mgr, sound_per_file=sound_per_file, seed=42)
        self.model = load_model_from_checkpoint(self.checkpoint, device=device)
        self.extractor = FeatureExtractor(self.model, batch_size=512, device=device)
        self.cache_dir.mkdir(parents=True, exist_ok=True) if self.cache_dir else None

        self.requests: List[Dict[str, Any]] = []
        self.results: Dict[str, np.ndarray] = {}
        self.cache_meta: Dict[str, Any] = {}

        self.transfer_results: Dict[str, Any] = {}
        self.deltaz_results: Dict[str, Any] = {}
        self.domain_clf_results: Dict[str, Any] = {}
        self.mlp_results: Dict[str, Any] = {}
        self.alignment_results: Dict[str, float] = {}
        self.canonical_results: Dict[str, Any] = {}

        self._build_sample_index()

    def _build_sample_index(self) -> None:
        self.requests = self.sampler.sample_all(self.manifest.files)


    # ------------------------------------------------------------------ #
    # Feature ingestion
    # ------------------------------------------------------------------ #
    def _file_grid(self, file_path: str) -> np.ndarray:
        """Standardized [270, 270, C] C-scan grid for a TDMS file."""
        return load_cscan_from_tdms(
            file_path,
            time_samples=500,
            temporal_samples=128,
            resample_mode="linear",
            normalization="global_peak",
            raster_correction=True,
            sX=300,
            sY=300,
            crop_border=15,
            apply_lowpass=True,
            lowpass_cutoff=2500.0,
            lowpass_order=4,
            standardize_coords=True,
        )

    def extract_all(self, progress: bool = True) -> None:
        """Run feature extraction for every sampled patch (cache to disk)."""
        cache = self.cache_dir / "features.json"
        if cache.exists():
            with open(cache, "r") as f:
                self.cache_meta = json.load(f)
            return

        results = self.extractor.extract_requests(self.requests, self._file_grid, progress=progress)
        self.results = results
        self.cache_meta = {
            "n_patches": int(len(results["pooled"])),
            "file_ids": [str(x) for x in results["file_id"]],
            "file_paths": [str(x) for x in results["file_path"]],
        }
        with open(cache, "w") as f:
            json.dump(self.cache_meta, f, indent=2)

    def feature_matrix(
        self,
        version: str,
        waveform: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, file_ids, patch_ids) for one latent version.

        y: 1 = defect, 0 = sound. Optional `waveform` filter by file waveform.
        """
        X = self.results[f"pooled_{version}"]
        y = np.zeros(X.shape[0], dtype=np.float32)
        paths = [str(p) for p in self.results["file_path"]]
        for i, p in enumerate(paths):
            if p in self.manifest._meta_by_path:
                m = self.manifest._meta_by_path[p]
                if m["kind"] == "defect":
                    y[i] = 1.0
        if waveform is not None:
            mask = np.array(
                [self.manifest._meta_by_path[p]["waveform"] == waveform for p in paths],
                dtype=bool,
            )
            X, y = X[mask], y[mask]
        return X, y, self.results["file_id"], self.results["file_id"]


    # ------------------------------------------------------------------ #
    # Experiment A: full 3x3 waveform transfer matrix (4 latent versions)
    # ------------------------------------------------------------------ #
    def run_experiment_a(
        self,
        train_waveforms: Sequence[str] = ("Square", "Gaussian", "Chirp"),
        test_waveforms: Sequence[str] = ("Square", "Gaussian", "Chirp"),
        classifier: str = "logreg",
    ) -> Dict[str, Any]:
        """3x3 transfer matrix per latent version.

        Train probe on files of `train_waveform`, evaluate on files of each
        `test_waveform`.  Results[version][train_wf][test_wf] = {auc, ap, ...}
        """
        versions = [
            NormalizationTransformer.NAME_RAW,
            NormalizationTransformer.NAME_CENTER,
            NormalizationTransformer.NAME_STD,
            NormalizationTransformer.NAME_WHITEN,
        ]
        train_waveforms = list(train_waveforms)

        paths = [str(p) for p in self.results["file_path"]]
        all_X = self.results["pooled_raw"]
        tr_mask = np.array(
            [p in self.manifest._meta_by_path and self.manifest._meta_by_path[p]["waveform"] in train_waveforms for p in paths],
            dtype=bool,
        )
        stats = NormalizationTransformer(all_X[tr_mask])

        self.transfer_results = {v: {} for v in versions}
        for tr_wf in train_waveforms:
            self.transfer_results[versions[0]][tr_wf] = {}
            X_tr_raw, y_tr, _, _ = self.feature_matrix(NormalizationTransformer.NAME_RAW, waveform=tr_wf)
            if len(X_tr_raw) == 0:
                for te_wf in test_waveforms:
                    self.transfer_results[versions[0]][tr_wf][te_wf] = {"auc": float("nan"), "ap": float("nan")}
                continue

            for te_wf in test_waveforms:
                X_te_raw, y_te, _, _ = self.feature_matrix(NormalizationTransformer.NAME_RAW, waveform=te_wf)
                if len(X_te_raw) == 0:
                    self.transfer_results[versions[0]][tr_wf][te_wf] = {"auc": float("nan"), "ap": float("nan")}
                    continue

                for ver in versions:
                    Xt = X_tr_raw
                    Xv = X_te_raw
                    if ver != NormalizationTransformer.NAME_RAW:
                        Xt = stats.transform(X_tr_raw, ver)
                        Xv = stats.transform(X_te_raw, ver)

                    if classifier == "logreg":
                        clf = fit_linear_probe(Xt, y_tr.astype(int), C=1.0, seed=42)
                    else:
                        clf = fit_mlp_probe(Xt, y_tr.astype(int), hidden=(64,), seed=42)
                    proba = clf.predict_proba(Xv)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xv)
                    auc = roc_auc_score(y_te.astype(int), proba)
                    ap = average_precision_score(y_te.astype(int), proba) if len(np.unique(y_te)) > 1 else float("nan")
                    self.transfer_results[ver][tr_wf][te_wf] = {
                        "auc": float(auc), "ap": float(ap),
                        "n_train": int(len(Xt)), "n_test": int(len(Xv)),
                    }
        return self.transfer_results


    # ------------------------------------------------------------------ #
    # Experiment B: defect-effect vectors + spatial-local contrast
    # ------------------------------------------------------------------ #
    def run_experiment_b(self, version: str = "raw") -> Dict[str, Any]:
        """Defect-effect vectors and spatial-local contrast vectors.

        pooled_delta : E[z|defect,w] - E[z|normal,w]  per waveform
        tile_delta   : per-tile spatial-local Delta z(p) = z(p) - mean(z(N(p)))
        cosine_matrix: per-tile-index cosine between Waveforms (mean over the
                       available defect patches), plus a tiled heatmap array.
        """
        ver = version
        paths = [str(p) for p in self.results["file_path"]]

        # --- pooled defect-effect vectors (128-dim foundation rep) --- #
        pooled_delta: Dict[str, np.ndarray] = {}
        for wf in self.waveforms:
            X = self.results[f"pooled_{ver}"]
            mask_def = np.array([self.manifest._meta_by_path[p]["kind"] == "defect" for p in paths], dtype=bool)
            X_def = X[mask_def]
            mask_norm = np.array([self.manifest._meta_by_path[p]["kind"] == "sound" for p in paths], dtype=bool)
            X_norm = X[mask_norm]
            if len(X_def) > 0 and len(X_norm) > 0:
                pooled_delta[wf] = X_def.mean(axis=0) - X_norm.mean(axis=0)
        out: Dict[str, Any] = {
            "pooled_delta": {wf: (v.astype(np.float32) if v is not None else None) for wf, v in pooled_delta.items()}
        }

        # --- per-tile spatial-local deviation per defect patch --- #
        tile_delta: Dict[str, List[np.ndarray]] = {wf: [] for wf in self.waveforms}
        for i, req in enumerate(self.requests):
            if self.results["kind"][i] != "defect":
                continue
            wf = self.manifest._meta_by_path[str(req["file_path"])]["waveform"]
            tile = self.results["tile"][i]
            dz = tile - tile.mean(axis=0, keepdims=True)
            tile_delta[wf].append(dz)

        mean_tile_delta = {wf: (np.mean(np.vstack(v), axis=0) if v else None) for wf, v in tile_delta.items()}

        n_tiles = self.results["tile"].shape[-1] if len(self.results["tile"]) else 25
        per_tile_cos: Dict[int, Dict[str, Dict[str, float]]] = {
            p: {fw: {tw: [] for tw in self.waveforms} for fw in self.waveforms} for p in range(n_tiles)
        }
        for i, req in enumerate(self.requests):
            if self.results["kind"][i] != "defect":
                continue
            wf = self.manifest._meta_by_path[str(req["file_path"])]["waveform"]
            dzp = self.results["tile"][i] - self.results["tile"][i].mean(axis=0, keepdims=True)  # [25,64]
            for p in range(n_tiles):
                dp = dzp[p]
                for tw in self.waveforms:
                    for j, req2 in enumerate(self.requests):
                        if j == i or self.results["kind"][j] != "defect":
                            continue
                        wf2 = self.manifest._meta_by_path[str(req2["file_path"])]["waveform"]
                        if wf2 != tw:
                            continue
                        if req["row"] != req2["row"] or req["col"] != req2["col"]:
                            continue
                        dzp2 = self.results["tile"][j] - self.results["tile"][j].mean(axis=0, keepdims=True)
                        dp2 = dzp2[p]
                        cos = float(np.dot(dp, dp2) / (np.linalg.norm(dp) * np.linalg.norm(dp2) + 1e-8))
                        per_tile_cos[p][wf][wf2].append(cos)

        cos_summary: Dict[str, Dict[str, float]] = {fw: {tw: 0.0 for tw in self.waveforms} for fw in self.waveforms}
        for p, pw in per_tile_cos.items():
            for fw in self.waveforms:
                for tw in self.waveforms:
                    v = pw[fw][tw]
                    if v:
                        cos_summary[fw][tw] += float(np.mean(v)) / n_tiles

        out["tile_delta"] = {wf: np.asarray(v, dtype=np.float32) if v else None for wf, v in tile_delta.items()}
        out["mean_tile_delta"] = mean_tile_delta
        out["per_tile_cosine"] = per_tile_cos
        out["cosine_matrix"] = cos_summary
        return out


    # ------------------------------------------------------------------ #
    # Experiment C: waveform domain classifier
    # ------------------------------------------------------------------ #
    def _file_level_stratified_split(
        self, test_size: float = 0.3, random_state: int = 42
    ) -> Tuple[Set[str], Set[str]]:
        """Split files (not patches) stratified by waveform."""
        from sklearn.model_selection import StratifiedShuffleSplit

        file_waveforms = {
            str(fp): self.manifest._meta_by_path[str(fp)]["waveform"] for fp in self.manifest.files
        }
        files = list(file_waveforms.keys())
        y_files = np.array([WAVEFORMS.index(file_waveforms[f]) for f in files])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(sss.split(files, y_files))
        return {files[i] for i in train_idx}, {files[i] for i in test_idx}

    def run_experiment_c(
        self,
        version: str = "raw",
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Waveform-domain classifier z -> waveform ID (file-level split)."""
        train_files, test_files = self._file_level_stratified_split(test_size, random_state)

        rows = []
        for i, req in enumerate(self.requests):
            fp = str(req["file_path"])
            if fp in train_files:
                rows.append((i, True))
            elif fp in test_files:
                rows.append((i, False))
        train_idx = [i for i, t in rows if t]
        test_idx = [i for i, t in rows if not t]

        X_tr = self.results[f"pooled_{version}"][train_idx]
        y_tr = np.array([self.manifest._meta_by_path[str(self.requests[i]["file_path"])]["waveform"] for i in train_idx])
        X_te = self.results[f"pooled_{version}"][test_idx]
        y_te = np.array([self.manifest._meta_by_path[str(self.requests[i]["file_path"])]["waveform"] for i in test_idx])

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_te_enc = le.transform(y_te)

        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=4000, multi_class="auto", random_state=random_state)
        clf.fit(X_tr, y_tr_enc)
        pred = clf.predict(X_te)

        self.domain_clf_results[version] = {
            "accuracy": float(accuracy_score(y_te_enc, pred)),
            "balanced_accuracy": float(
                sum(np.diag(confusion_matrix(y_te_enc, pred)) / np.bincount(y_te_enc, minlength=len(le.classes_))) / len(le.classes_)
                if len(le.classes_) else float("nan")
            ),
            "n_test": int(len(y_te)),
            "classes": le.classes_.tolist(),
            "confusion_matrix": confusion_matrix(y_te_enc, pred).tolist(),
        }
        return self.domain_clf_results[version]


    # ------------------------------------------------------------------ #
    # Experiment D: linear vs small MLP on cross-waveform transfer
    # ------------------------------------------------------------------ #
    def run_experiment_d(
        self,
        train_waveform: str = "Square",
        test_waveform: str = "Chirp",
        mlp_configs: Sequence[Tuple[str, Tuple[int, ...]]] = (("mlp_64", (64,)), ("mlp_32_32", (32, 32))),
    ) -> Dict[str, Any]:
        """Linear vs small MLP probe on a cross-waveform transfer direction.

        Binary defect detection; trained on `train_waveform`, evaluated on
        `test_waveform`.  Returns AUC/AP/accuracy for linear and each MLP.
        """
        X_tr_raw, y_tr, _, _ = self.feature_matrix(NormalizationTransformer.NAME_RAW, waveform=train_waveform)
        X_te_raw, y_te, _, _ = self.feature_matrix(NormalizationTransformer.NAME_RAW, waveform=test_waveform)
        if len(X_tr_raw) == 0 or len(X_te_raw) == 0:
            return {"error": "empty train/test set for this transfer direction"}

        paths = [str(p) for p in self.results["file_path"]]
        tr_mask = np.array(
            [p in self.manifest._meta_by_path and self.manifest._meta_by_path[p]["waveform"] == train_waveform for p in paths],
            dtype=bool,
        )
        te_mask = np.array(
            [p in self.manifest._meta_by_path and self.manifest._meta_by_path[p]["waveform"] == test_waveform for p in paths],
            dtype=bool,
        )

        results = {"linear": None, "mlp": {}}
        if tr_mask.sum() > 0 and te_mask.sum() > 0:
            stats = NormalizationTransformer(X_tr_raw)
            X_tr = stats.transform(X_tr_raw, NormalizationTransformer.NAME_STD)
            X_te = stats.transform(X_te_raw, NormalizationTransformer.NAME_STD)
            y_tr_i = y_tr.astype(int)
            y_te_i = y_te.astype(int)

            lin = LogisticRegression(C=1.0, solver="liblinear", max_iter=4000, class_weight="balanced", random_state=42)
            lin.fit(X_tr, y_tr_i)
            proba_l = lin.predict_proba(X_te)[:, 1]
            results["linear"] = {
                "auc": float(roc_auc_score(y_te_i, proba_l)),
                "ap": float(average_precision_score(y_te_i, proba_l)),
                "accuracy": float(accuracy_score(y_te_i, lin.predict(X_te))),
                "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            }

            for name, hidden in mlp_configs:
                mlp = fit_mlp_probe(X_tr, y_tr_i, hidden=hidden, seed=42)
                proba_m = mlp.predict_proba(X_te)[:, 1] if hasattr(mlp, "predict_proba") else mlp.predict(X_te)
                results["mlp"][name] = {
                    "auc": float(roc_auc_score(y_te_i, proba_m)),
                    "ap": float(average_precision_score(y_te_i, proba_m)),
                    "accuracy": float(accuracy_score(y_te_i, (proba_m >= 0.5).astype(int))),
                    "hidden": list(hidden),
                    "n_iters": int(mlp.n_iter_[0]),
                }
        self.mlp_results[f"{train_waveform}_to_{test_waveform}"] = results
        return results


    # ------------------------------------------------------------------ #
    # Experiment E: progressive alignment curve
    # ------------------------------------------------------------------ #
    def run_experiment_e(
        self,
        train_waveform: str = "Square",
        test_waveform: str = "Chirp",
    ) -> Dict[str, Any]:
        """Progressive alignment: Raw -> Center -> Standardize -> Whiten -> CORAL.

        At every step a binary defect probe is trained on the transformed
        `train_waveform` features and evaluated on the transformed `test_waveform`
        features.  CORAL is the only step that uses target (test) statistics.
        """
        X_tr_raw, y_tr, _, _ = self.feature_matrix(NormalizationTransformer.NAME_RAW, waveform=train_waveform)
        X_te_raw, y_te, _, _ = self.feature_matrix(NormalizationTransformer.NAME_RAW, waveform=test_waveform)
        if len(X_tr_raw) == 0 or len(X_te_raw) == 0:
            return {"error": "empty set"}

        y_tr_i = y_tr.astype(int)
        y_te_i = y_te.astype(int)

        stats = NormalizationTransformer(X_tr_raw)
        steps = {
            "raw": X_tr_raw,
            "center": stats.transform(X_tr_raw, NormalizationTransformer.NAME_CENTER),
            "standardize": stats.transform(X_tr_raw, NormalizationTransformer.NAME_STD),
            "whitened": stats.transform(X_tr_raw, NormalizationTransformer.NAME_WHITEN),
        }
        # CORAL: source is train, target is test -> moment matching
        steps["coral"] = NormalizationTransformer.coral(X_tr_raw, X_te_raw)
        # test side stays in the original test space for evaluation
        steps["coral_test"] = X_te_raw

        results = {}
        for name, Xtr in steps.items():
            if name == "coral":
                continue
            Xte = stats.transform(X_te_raw, name)
            clf = fit_linear_probe(Xtr, y_tr_i, C=1.0, seed=42)
            proba = clf.predict_proba(Xte)[:, 1]
            results[name] = {"auc": float(roc_auc_score(y_te_i, proba)),
                             "ap": float(average_precision_score(y_te_i, proba))}

        # CORAL step
        clf = fit_linear_probe(steps["coral"], y_tr_i, C=1.0, seed=42)
        proba_c = clf.predict_proba(steps["coral_test"])[:, 1]
        results["coral"] = {"auc": float(roc_auc_score(y_te_i, proba_c)),
                            "ap": float(average_precision_score(y_te_i, proba_c))}

        self.alignment_results = results
        return results


    # ------------------------------------------------------------------ #
    # Experiment F: physics canonicalization baseline
    # ------------------------------------------------------------------ #
    def _estimate_square_template(self, n_samples: int = 200) -> np.ndarray:
        """Average of raw Square waveforms in the corpus (measurement excitation)."""
        square_files = [
            str(fp) for fp in self.manifest.files
            if self.manifest._meta_by_path[str(fp)]["waveform"] == "Square"
        ][:n_samples]
        if not square_files:
            return np.ones(500, dtype=np.float32)
        waves = []
        for fp in square_files:
            raw = read_tdms_1d_waveforms(fp, target_time_samples=500, normalization="none")
            waves.append(raw)
        stack = np.vstack(waves)
        return stack.mean(axis=0).astype(np.float32)

    def _canonical_grid_loader(
        self, canonical_type: str
    ) -> Callable[[str], np.ndarray]:
        """Return a grid loader that applies the physics canonicalization."""
        template = self._estimate_square_template()

        def build_loader(envelope_only: bool):
            def loader(file_path: str) -> np.ndarray:
                raw = read_tdms_1d_waveforms(file_path, target_time_samples=500, normalization="none")  # [N, 500]
                if envelope_only:
                    # F1: mean-normalised RMS envelope (pulse-compressed canonical form)
                    canon = moving_rms_envelope(raw, win=51, eps=1e-8)
                else:
                    # F2: Wiener matched-filter deconvolution by the square template
                    S = np.fft.rfft(raw, axis=1)
                    P = np.fft.rfft(template, axis=0)[np.newaxis, :]
                    H = np.conj(P) * S / (np.abs(P) ** 2 + 1e-6)
                    canon = np.fft.irfft(H, n=500, axis=1).astype(np.float32)
                    canon = np.abs(canon)  # stable magnitude scale
                canon = linear_time_resample(canon, n_out=128)
                canon = apply_lowpass_filter(canon, cutoff_hz=2500.0, fs=25600.0, order=4)
                canon = normalize_waveforms_linear(canon, normalization="file_peak")
                grid = canon[: 300 * 300].reshape(300, 300, 128)
                try:
                    from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
                    gt_mgr = get_ground_truth_manager()
                    grid = gt_mgr.transform_cscan_to_standard(grid, file_path)
                except Exception:
                    pass
                return grid.astype(np.float32)
            return loader

        if canonical_type == "envelope":
            return build_loader(envelope_only=True)
        if canonical_type == "wiener":
            return build_loader(envelope_only=False)
        raise ValueError(f"Unknown canonicalization: {canonical_type}")

    def canonicalize_features(self, canonical_type: str) -> Dict[str, Any]:
        """Zero-shot transfer on canonicalized grids (same frozen encoder)."""
        loader = self._canonical_grid_loader(canonical_type)
        results = self.extractor.extract_requests(self.requests, loader, progress=False)

        paths = [str(p) for p in self.results["file_path"]]
        tr_mask = np.array(
            [p in self.manifest._meta_by_path and self.manifest._meta_by_path[p]["waveform"] == "Square" for p in paths],
            dtype=bool,
        )
        te_mask = np.array(
            [p in self.manifest._meta_by_path and self.manifest._meta_by_path[p]["waveform"] == "Chirp" for p in paths],
            dtype=bool,
        )

        def patch_labels(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
            idx = np.where(mask)[0]
            y = np.zeros(len(mask), dtype=np.float32)
            for k, i in enumerate(idx):
                fp = str(self.requests[i]["file_path"])
                if fp in self.manifest._meta_by_path and self.manifest._meta_by_path[fp]["kind"] == "defect":
                    y[k] = 1.0
            return y

        stats = NormalizationTransformer(self.results["pooled_raw"][tr_mask])
        y_tr = patch_labels(self.results["pooled_raw"], tr_mask).astype(int)
        y_te = patch_labels(self.results["pooled_raw"], te_mask).astype(int)

        X_tr_r = stats.transform(self.results["pooled_raw"][tr_mask], NormalizationTransformer.NAME_STD)
        X_te_r = stats.transform(self.results["pooled_raw"][te_mask], NormalizationTransformer.NAME_STD)
        lin_r = LogisticRegression(C=1.0, solver="liblinear", max_iter=4000, class_weight="balanced", random_state=42)
        lin_r.fit(X_tr_r, y_tr)
        base_auc = float(roc_auc_score(y_te, lin_r.predict_proba(X_te_r)[:, 1]))

        X_tr_c = stats.transform(results["pooled_raw"][tr_mask], NormalizationTransformer.NAME_STD)
        X_te_c = stats.transform(results["pooled_raw"][te_mask], NormalizationTransformer.NAME_STD)
        lin_c = LogisticRegression(C=1.0, solver="liblinear", max_iter=4000, class_weight="balanced", random_state=42)
        lin_c.fit(X_tr_c, y_tr)
        can_auc = float(roc_auc_score(y_te, lin_c.predict_proba(X_te_c)[:, 1]))

        self.canonical_results[canonical_type] = {
            "n_patches": int(len(results["pooled_raw"])),
            "baseline_auc": base_auc,
            "canonical_auc": can_auc,
            "delta_auc": can_auc - base_auc,
            "mean_abs_feature": float(np.mean(np.abs(results["pooled_raw"]))),
        }
        return self.canonical_results[canonical_type]


    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    def plot_transfer_matrix(
        self, results: Dict[str, Any], version: str = "raw", save_path: Optional[str] = None
    ) -> None:
        """Heatmap of the 3x3 transfer matrix for one latent version."""
        train_wfs = list(results[version].keys())
        test_wfs = list(results[version][train_wfs[0]].keys())
        mat = np.array(
            [[results[version][tw][te]["auc"] for te in test_wfs] for tw in train_wfs]
        )
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(test_wfs)))
        ax.set_xticklabels(test_wfs)
        ax.set_yticks(range(len(train_wfs)))
        ax.set_yticklabels(train_wfs)
        ax.set_xlabel("Test waveform")
        ax.set_ylabel("Train waveform")
        ax.set_title(f"Transfer AUC (linear probe, Raw) — exp28 5x5")
        for i in range(len(train_wfs)):
            for j in range(len(test_wfs)):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, label="AUC")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            print("  saved:", save_path)
        plt.close(fig)

    def plot_cosine_heatmap(
        self,
        cos_matrix: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Heatmap of pairwise cosine between defect-effect vectors."""
        wfs = list(cos_matrix.keys())
        mat = np.array(
            [[cos_matrix[fw][tw] if cos_matrix[fw].get(tw) is not None else np.nan for tw in wfs] for fw in wfs]
        )
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(wfs)))
        ax.set_xticklabels(wfs)
        ax.set_yticks(range(len(wfs)))
        ax.set_yticklabels(wfs)
        ax.set_xlabel("Waveform")
        ax.set_ylabel("Waveform")
        ax.set_title("Mean cosine(Δz) between defect-effect vectors")
        for i in range(len(wfs)):
            for j in range(len(wfs)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "—", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, label="cosine")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            print("  saved:", save_path)
        plt.close(fig)


    def plot_alignment_curve(self, results: Dict[str, float], save_path: Optional[str] = None) -> None:
        """Progressive alignment curve (Experiment E)."""
        order = ["raw", "center", "standardize", "whitened", "coral"]
        labels = {
            "raw": "Raw", "center": "Center", "standardize": "Standardize",
            "whitened": "Whiten", "coral": "CORAL",
        }
        aucs = [results.get(s, {}).get("auc", np.nan) for s in order]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(range(len(order)), aucs, "o-", color="#1f77b4", linewidth=2, markersize=7)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[s] for s in order])
        ax.set_xlabel("Alignment step")
        ax.set_ylabel("Zero-shot AUC (Square→Chirp)")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Progressive alignment → probe AUC")
        for x, v in zip(range(len(order)), aucs):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            print("  saved:", save_path)
        plt.close(fig)

    def plot_latent_tsne(
        self, version: str = "raw", sample: int = 3000, save_path: Optional[str] = None
    ) -> None:
        """t-SNE of pooled features colored by waveform."""
        from sklearn.manifold import TSNE

        X = self.results[f"pooled_{version}"]
        y = np.array([self.manifest._meta_by_path[str(p)]["waveform"] for p in self.results["file_path"]])
        rng = np.random.default_rng(42)
        if len(X) > sample:
            idx = rng.choice(len(X), size=sample, replace=False)
            X, y = X[idx], y[idx]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
        Z = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(7.5, 6))
        colors = {"Square": "#1f77b4", "Gaussian": "#2ca02c", "Chirp": "#ff7f0e"}
        for wf, col in colors.items():
            m = y == wf
            ax.scatter(Z[m, 0], Z[m, 1], c=col, label=wf, s=6, alpha=0.55, edgecolors="none")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE of latent features (Waveform label) — {version}")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            print("  saved:", save_path)
        plt.close(fig)

    def run_all(
        self,
        out_dir: Optional[str] = None,
        eval_a: bool = True,
        eval_b: bool = True,
        eval_c: bool = True,
        eval_d: bool = True,
        eval_e: bool = True,
        eval_f: bool = True,
        eval_tsne: bool = True,
    ) -> Dict[str, Any]:
        """Run the full audit suite and save JSON + figures."""
        out_dir = Path(out_dir) if out_dir else self.cache_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        self.extract_all(progress=True)

        if eval_a:
            print("[A] 3x3 transfer matrix ...", flush=True)
            self.run_experiment_a()
            for ver in self.transfer_results:
                self.plot_transfer_matrix(self.transfer_results, version=ver, save_path=str(out_dir / f"transfer_matrix_{ver}.png"))

        if eval_b:
            print("[B] defect-effect vectors ...", flush=True)
            self.deltaz_results = self.run_experiment_b(version="raw")
            self.plot_cosine_heatmap(self.deltaz_results["cosine_matrix"], save_path=str(out_dir / "deltaz_cosine_heatmap.png"))

        if eval_c:
            print("[C] waveform domain classifier ...", flush=True)
            self.run_experiment_c()

        if eval_d:
            print("[D] linear vs MLP ...", flush=True)
            self.run_experiment_d(train_waveform="Square", test_waveform="Chirp")

        if eval_e:
            print("[E] progressive alignment ...", flush=True)
            self.run_experiment_e(train_waveform="Square", test_waveform="Chirp")
            self.plot_alignment_curve(self.alignment_results, save_path=str(out_dir / "alignment_curve.png"))

        if eval_f:
            print("[F] physics canonicalization ...", flush=True)
            self.canonicalize_features("wiener")
            self.canonicalize_features("envelope")

        if eval_tsne:
            self.plot_latent_tsne(version="raw", sample=4000, save_path=str(out_dir / "tsne_waveform.png"))

        summary = {
            "manifest": self.manifest.to_dict(),
            "transfer": self.transfer_results,
            "deltaz": self.deltaz_results,
            "domain_clf": self.domain_clf_results,
            "mlp": self.mlp_results,
            "alignment": self.alignment_results,
            "canonical": self.canonical_results,
        }
        with open(out_dir / "latent_geometry_audit_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
        print("Saved summary to", out_dir / "latent_geometry_audit_summary.json")
        return summary


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #
def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Latent Geometry Audit for PECT-JEPA (frozen EXP-28).")
    parser.add_argument("--checkpoint", default="experiments/5x5/exp28_full_20ep/checkpoints/best_model_5x5.pt")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--specimen", default="Corrosion")
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--output_dir", default="experiments/5x5/exp28_full_20ep/latent_geometry_audit")
    parser.add_argument("--no_a", action="store_true")
    parser.add_argument("--no_b", action="store_true")
    parser.add_argument("--no_c", action="store_true")
    parser.add_argument("--no_d", action="store_true")
    parser.add_argument("--no_e", action="store_true")
    parser.add_argument("--no_f", action="store_true")
    parser.add_argument("--no_tsne", action="store_true")
    args = parser.parse_args(argv)

    audit = LatentGeometryAudit(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        specimen=args.specimen,
        sensor=args.sensor,
        sound_per_file=8,
        cache_dir=args.output_dir,
    )
    print(f"Loaded model from {args.checkpoint} on {args.device}", flush=True)
    print(f"Manifest: {len(audit.manifest)} files | waveforms={audit.manifest.counts}", flush=True)
    summary = audit.run_all(
        out_dir=args.output_dir,
        eval_a=not args.no_a,
        eval_b=not args.no_b,
        eval_c=not args.no_c,
        eval_d=not args.no_d,
        eval_e=not args.no_e,
        eval_f=not args.no_f,
        eval_tsne=not args.no_tsne,
    )
    return summary


if __name__ == "__main__":
    main()

