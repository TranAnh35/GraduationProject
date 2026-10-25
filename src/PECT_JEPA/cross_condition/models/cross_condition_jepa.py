"""
Cross-Condition Paired JEPA Architecture for PECT.

Single canonical pipeline:
- Inputs: pair (x_A, x_B) at identical physical coordinates from two measurement conditions.
- Condition token: Dodd-Deeds spectral signature c_B extracted from the sound-metal
  background patch ref_B of Condition B (0% DC leakage).
- Masking: context from a contiguous non-enclosing wedge of Condition A, target from the
  complementary wedge of Condition B.
- Objective: predict H_tgt(B) from H_ctx(A) conditioned on c_B, plus VICReg rank guards
  and an optional background-gated cross-condition invariance term.
- Representation: unified 64D latent, pooled as Z = [h_center, mean(h_context)] in R^128.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.config import CrossConditionConfig
from ..data.topologies import get_contiguous_star_masks
from .tokenizer import DualDomainTokenizer
from .context_encoder import ContextEncoder
from .target_encoder import TargetEncoder
from .predictor import CrossConditionPredictor


# Dodd-Deeds spectral condition token: number of harmonic bins kept above DC.
SPECTRAL_N_HARMONICS = 8


def spectral_token_dim(config=None) -> int:
    """Feature width of the spectral condition signature.

    Each harmonic contributes (cos theta_k, sin theta_k) weighted by sqrt(E_k/sum E)
    -- continuous at the +-pi branch cut -- plus its normalized energy ratio.
    """
    return 3 * SPECTRAL_N_HARMONICS


def corner_background_energy(cscan: np.ndarray) -> float:
    """Median per-probe temporal variance of the four defect-free corner patches.

    Mirrors PairedPECTDataset's corner-median reference patch so the global
    background anchor of the tokenizer is identical at train and inference time.
    """
    c = np.asarray(cscan, dtype=np.float32)
    corners = np.stack([c[0:5, 0:5, :], c[0:5, -5:, :], c[-5:, 0:5, :], c[-5:, -5:, :]], axis=0)
    ref = np.median(corners, axis=0).reshape(-1, c.shape[-1])
    return float(np.median(ref.std(axis=-1) ** 2))


def normalized_probe_l1(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Probe-wise normalized L1: z-score each probe over its feature dimension.

    Invariant to any per-probe rescaling of the representation, so shrinking the
    latent cannot reduce the loss (removes the death-spiral descent direction of
    raw L1), while a position-blind / constant prediction is still penalized.

    Computed in float32: under AMP the std can be a float16 value small enough
    that `clamp_min(1e-8)` rounds to 0.0, producing inf/NaN and skipped steps
    (exp71a epoch 35-52). `clamp_min(1e-6)` is also representable in fp16.
    """
    pred_f = pred.float()
    tgt_f = tgt.float()
    pred_m = pred_f.mean(dim=-1, keepdim=True)
    pred_s = pred_f.std(dim=-1, keepdim=True).clamp_min(1e-6)
    h_pred_norm = (pred_f - pred_m) / pred_s

    tgt_m = tgt_f.mean(dim=-1, keepdim=True)
    tgt_s = tgt_f.std(dim=-1, keepdim=True).clamp_min(1e-6)
    h_tgt_norm = (tgt_f - tgt_m) / tgt_s
    return (h_pred_norm - h_tgt_norm).abs().mean(dim=(-2, -1)).mean()


def center_by_group(
    z_f: torch.Tensor,
    group_id: Optional[torch.Tensor] = None,
    min_group_size: int = 2,
) -> torch.Tensor:
    """Center a 2D tensor z_f [N, D] per group_id [N] (global mean if None).

    Groups with fewer than `min_group_size` elements in the batch fall back to
    the global batch mean. This is critical for the probe-level VICReg: a single
    patch contributes 16 probes, so a group of `counts <= 16` is one window and
    subtracting its own probe mean would erase the spatial (r, c) variance that
    VICReg is supposed to protect (exp71: val_sound_var -> 1.6e-5 while R_eff
    stayed at 61.9D on static positional offsets).

    Always computed in float32: the per-group mean buffer below is float32, so a
    float16 input would silently upcast anyway; the explicit cast also keeps the
    group statistics out of the AMP fp16 range.
    """
    z_f = z_f.float()
    if group_id is None:
        return z_f - z_f.mean(dim=0, keepdim=True)
    uniq, inv = torch.unique(group_id, return_inverse=True)  # inv: [N] in 0..S-1
    n_groups = uniq.numel()
    counts = torch.bincount(inv, minlength=n_groups)  # [S] int64
    sums = torch.zeros(n_groups, z_f.shape[1], device=z_f.device, dtype=torch.float32)
    sums.index_add_(0, inv, z_f)
    means = sums / counts.clamp_min(1).unsqueeze(1).float()  # [S, D]
    m = means[inv]  # [N, D] float32
    small_group = (counts[inv] < int(min_group_size)).unsqueeze(1)  # [N, 1]
    global_mean = z_f.mean(dim=0, keepdim=True)
    return torch.where(small_group, z_f - global_mean, z_f - m)


def compute_vicreg_loss(
    z: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
    scan_id: Optional[torch.Tensor] = None,
    min_group_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg coordinate-wise variance hinge and covariance decorrelation penalty.

    If scan_id is provided, z is centered per scan before computing variance and
    covariance, so between-scan drift cannot falsely satisfy the variance
    regularizer; the model must amplify true within-scan flaw variance.
    Groups smaller than `min_group_size` fall back to the global mean instead of
    being left uncentered (a singleton group must never satisfy the hinge by
    keeping its own mean).
    """
    z_f = z.float()
    z_cent = center_by_group(z_f, scan_id, min_group_size=min_group_size)
    std = torch.sqrt(z_cent.var(dim=0, unbiased=False) + eps)

    var_loss = torch.mean(F.relu(gamma - std))

    N = z_cent.shape[0]
    cov = (z_cent.T @ z_cent) / max(N - 1, 1)
    cov_off = cov - torch.diag(torch.diag(cov))
    cov_loss = (cov_off ** 2).sum() / z_f.shape[1]

    total = var_weight * var_loss + cov_weight * cov_loss
    return total, var_loss, cov_loss


class CrossConditionJEPA(nn.Module):
    """Self-supervised cross-condition paired JEPA model (zero metadata injection)."""

    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config

        # 1. Physical dual-domain tokenizer
        self.tokenizer = DualDomainTokenizer(config)

        # 2. Context encoder (Condition A) / 3. target encoder (Condition B, EMA teacher)
        self.context_encoder = ContextEncoder(config)
        self.target_encoder = TargetEncoder(config)
        self._init_target_encoder()

        # 4. Self-supervised condition projector (legacy c_B path, kept for parity)
        self.cond_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        # 4b. Dodd-Deeds spectral condition projector
        self.spec_cond_proj = nn.Sequential(
            nn.Linear(spectral_token_dim(), config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # 5. Cross-condition predictor
        self.predictor = CrossConditionPredictor(config)

        # 6. Asymmetric invariance predictor (BYOL-style). Absorbs the alignment
        # work of L_inv so the encoder itself is never pulled directly onto the
        # EMA target (FM6: direct symmetric L_inv collapses rank 23D -> 1.2D).
        if getattr(config, "inv_predictor", False):
            _D = config.embed_dim
            _H = max(2 * _D, 64)
            self.inv_head = nn.Sequential(
                nn.Linear(_D, _H),
                nn.GELU(),
                nn.Linear(_H, _D),
            )
        else:
            self.inv_head = None

        # Concentric star partitioning: indices 0..8 core (target), 9..24 outer (context)
        self.register_buffer("target_idx", torch.arange(config.num_target_probes, dtype=torch.long))
        self.register_buffer("context_idx", torch.arange(config.num_target_probes, 25, dtype=torch.long))

        # Contiguous non-enclosing wedge mask bank (16 masks: 8 outer half-plane
        # wedges with center 0 in context + 8 apex-inclusive wedges with center 0
        # in target). Non-persistent so checkpoints stay lean.
        _rays = getattr(config, "num_target_rays", 3)
        tgt_bank_np, ctx_bank_np = get_contiguous_star_masks(
            star_radii=tuple(getattr(config, "star_radii", (2, 4, 7))),
            num_target_rays=_rays,
        )
        self.register_buffer("target_idx_bank", torch.from_numpy(tgt_bank_np).long(), persistent=False)
        self.register_buffer("context_idx_bank", torch.from_numpy(ctx_bank_np).long(), persistent=False)

        # 7. EXP-72A: Green radial-decay / center-guided spatial aggregation.
        #    `probe_r2` is the normalized squared physical radius of each of the 25
        #    star probes (r1=2px -> 0.082, r2=4px -> 0.327, r3=7px -> 1.0), so the
        #    outer ring can be down-weighted instead of leaking a 15x15px false
        #    positive halo around every defect (FM13).
        self.spatial_pool_mode = str(getattr(self.config, "spatial_pool_mode", "mean"))
        _norm_coords = self.tokenizer.norm_coords.squeeze(0).float()   # [25, 2]
        _r2_raw = (_norm_coords ** 2).sum(dim=-1)                      # [25]
        self.register_buffer("probe_r2", _r2_raw / _r2_raw.max().clamp_min(1e-6), persistent=False)

        if self.spatial_pool_mode == "radial":
            init_alpha = float(max(0.1, getattr(self.config, "radial_decay_init", 2.5)))
            self.radial_log_alpha = nn.Parameter(
                torch.tensor(float(np.log(init_alpha)), dtype=torch.float32)
            )
            self.radial_attn_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=False)
            nn.init.normal_(self.radial_attn_proj.weight, std=1e-2)
        else:
            # No parameters in 'mean' mode: exp70/exp71 checkpoints load strict=True.
            self.radial_log_alpha = None
            self.radial_attn_proj = None

    def _pool_spatial_context(
        self,
        h_probes: torch.Tensor,
        probe_indices: torch.Tensor,
        h_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregates spatial context probe representations [B, N, D] -> [B, D].

        - 'mean' mode: uniform average across all N probes (legacy behavior).
        - 'radial' mode: physical Green radial-decay prior (-alpha * r_i^2) combined
          with center-guided content attention so inner-ring probes (r1=2px) dominate
          and far-field outer-ring probes (r3=7px) do not leak false-positive halos.
        """
        if self.spatial_pool_mode != "radial" or self.radial_log_alpha is None:
            return h_probes.mean(dim=1)

        B, N, D = h_probes.shape
        # Gather normalized squared radius for each probe in the subset: [B, N]
        if probe_indices.ndim == 1:
            r2 = self.probe_r2[probe_indices].unsqueeze(0).expand(B, -1)
        else:
            r2 = self.probe_r2[probe_indices]
        r2 = r2.to(dtype=torch.float32, device=h_probes.device)

        # Clamp alpha in [0.2, 10.0] for numerical stability
        alpha = torch.exp(self.radial_log_alpha.float()).clamp(min=0.2, max=10.0)
        radial_logits = -alpha * r2  # [B, N]

        if self.radial_attn_proj is not None:
            if h_anchor is None:
                # Training on 16 context probes: form the local anchor with pure
                # Green radial weights (there is no center probe in the window).
                w_base = torch.softmax(radial_logits, dim=-1).to(dtype=h_probes.dtype)
                h_anchor = torch.einsum("bn,bnd->bd", w_base, h_probes)
            q = self.radial_attn_proj(h_anchor.float()).unsqueeze(1).float()  # [B, 1, D]
            sim_logits = (q * h_probes.float()).sum(dim=-1) / (float(D) ** 0.5)  # [B, N]
            logits = radial_logits + torch.tanh(sim_logits)
        else:
            logits = radial_logits

        weights = torch.softmax(logits, dim=-1).to(dtype=h_probes.dtype)  # [B, N]
        return torch.einsum("bn,bnd->bd", weights, h_probes)

    def _init_target_encoder(self):
        """Initialize target encoder weights identical to the context encoder."""
        for p_tgt, p_ctx in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad = False

    def update_target_encoder(self, momentum: float = 0.996):
        """EMA update of the target encoder."""
        self.target_encoder.update_ema(self.context_encoder, momentum=momentum)

    def extract_spectral_condition_token(self, ref_b: torch.Tensor) -> torch.Tensor:
        """
        Extracts the Dodd-Deeds spectral signature c_B from the sound-metal patch ref_b.
        Eliminates DC amplitude and lift-off attenuation. Per harmonic k = 1..8:
          - w_k * cos theta_k, w_k * sin theta_k with w_k = sqrt(E_k / sum E),
            theta_k = arg(X_k) -- continuous at the +-pi branch cut, and bins with
            ~zero energy (arbitrary phase) contribute ~zero;
          - normalized spectral energy ratio E_k / sum(E).
        Args:
            ref_b: [B, 5, 5, C] raw background sound-metal patch.
        Returns:
            c_b: [B, 1, D] condition token.
        """
        B = ref_b.shape[0]
        C = ref_b.shape[-1]
        ref_flat = ref_b.view(B, -1, C)
        ref_mean = ref_flat.mean(dim=1)  # [B, C]
        ref_ac = ref_mean - ref_mean.mean(dim=-1, keepdim=True)  # remove DC
        X = torch.fft.rfft(ref_ac, dim=-1)  # [B, C // 2 + 1] complex
        X_harm = X[:, 1:1 + SPECTRAL_N_HARMONICS]
        mags = torch.abs(X_harm)
        energies = mags ** 2
        energy_ratios = energies / (energies.sum(dim=-1, keepdim=True) + 1e-8)

        theta = torch.angle(X_harm)  # [B, 8] in [-pi, pi]
        weight = torch.sqrt(energy_ratios.clamp_min(0.0))
        phase_feat = torch.stack(
            [weight * torch.cos(theta), weight * torch.sin(theta)], dim=-1
        ).reshape(B, -1)
        spec_feat = torch.cat([phase_feat, energy_ratios], dim=-1)
        return self.spec_cond_proj(spec_feat).unsqueeze(1)

    def extract_condition_token(self, ref_b: torch.Tensor) -> torch.Tensor:
        """Extracts c_B [B, 1, D] from the sound-metal background patch of Condition B."""
        return self.extract_spectral_condition_token(ref_b)

    @staticmethod
    def _bg_ref_energy(ref: torch.Tensor) -> torch.Tensor:
        """Median per-probe temporal variance of a reference patch -> [B, 1, 1]."""
        B = ref.shape[0]
        rf = ref.reshape(B, -1, ref.shape[-1]).float()
        return torch.median(rf.std(dim=-1) ** 2, dim=1, keepdim=True).values.unsqueeze(-1)

    @staticmethod
    def _probe_relative_energy(x: torch.Tensor) -> torch.Tensor:
        """Unsupervised per-probe spatial energy contrast -> [B, 25].

        Same quantity the tokenizer feeds the time branch as Delta E_local, used
        here only to decide which tokens are sound metal (no labels involved).
        """
        B = x.shape[0]
        xf = x.reshape(B, 25, -1).float()
        e_probe = xf.std(dim=-1) ** 2
        e_context = e_probe[:, 1:].mean(dim=1, keepdim=True) + 1e-8
        return torch.log1p(e_probe) - torch.log1p(e_context)

    def _background_gate(
        self,
        x_a: torch.Tensor,
        ctx_idx: torch.Tensor,
        batch_arange: torch.Tensor,
        keep_q: float,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Selects the lowest-|Delta E| tokens (sound metal) for L_inv.

        Defect-edge tokens carry large condition-specific scattering contrast; a
        blind L_inv over all tokens washes that boundary away. Gating is fully
        unsupervised (energy rank within the batch, no ground truth).
        Returns (flat bool mask over B*N_ctx or None, kept fraction).
        """
        if keep_q <= 0.0 or keep_q >= 1.0:
            return None, 1.0
        rel = self._probe_relative_energy(x_a)  # [B, 25]
        score = rel.gather(1, ctx_idx).abs().reshape(-1)
        if score.numel() < 8:
            return None, 1.0
        cut = torch.quantile(score, keep_q)
        keep = score <= cut
        kept = int(keep.sum())
        if kept < max(1, int(0.05 * score.numel())):
            return None, 1.0
        return keep, float(kept) / float(score.numel())

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        ref_b: torch.Tensor,
        ref_a: Optional[torch.Tensor] = None,
        scan_id: Optional[torch.Tensor] = None,
        scan_id_b: Optional[torch.Tensor] = None,
        inv_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass during self-supervised pretraining:
        x_a:     [B, 5, 5, C] measurement patch from Condition A
        x_b:     [B, 5, 5, C] measurement patch from Condition B (same physical coordinate)
        ref_b:   [B, 5, 5, C] sound-metal background patch of Condition B
        ref_a:   [B, 5, 5, C] sound-metal background patch of Condition A (optional)
        scan_id: [B] file id of Condition A for within-scan centered regularization
        scan_id_b: [B] file id of Condition B (defaults to scan_id)
        inv_scale: curriculum multiplier in [0, 1] for L_inv (default 1.0)
        """
        B = x_a.shape[0]
        device = x_a.device

        # 1. Dual-domain tokenization: each condition gets its OWN background
        #    anchor. Passing A's anchor into the B branch leaked A's background
        #    energy into the target side and made H_tgt_b depend on ref_a.
        bg_ref_a = None
        bg_ref_b = None
        if getattr(self.config, "global_bg_energy_anchor", False):
            bg_ref_a = self._bg_ref_energy(ref_a if ref_a is not None else ref_b)
            bg_ref_b = self._bg_ref_energy(ref_b if ref_b is not None else ref_a)
        tokens_a, pos_a = self.tokenizer(x_a, bg_ref_energy=bg_ref_a)  # [B, 25, D]
        tokens_b, pos_b = self.tokenizer(x_b, bg_ref_energy=bg_ref_b)

        # 2. Self-supervised condition token
        c_b = self.extract_condition_token(ref_b) if self.config.use_condition_token else None

        # 3. Partition tokens: contiguous non-enclosing wedges (default) or the
        #    fixed 16 outer context / 9 inner target split (ablation).
        batch_arange = torch.arange(B, device=device).unsqueeze(1)
        if getattr(self.config, "contiguous_masking", True):
            num_masks = self.target_idx_bank.shape[0]
            if self.training:
                mask_sel = torch.randint(0, num_masks, (B,), device=device)
            else:
                mask_sel = torch.arange(B, device=device) % num_masks
            ctx_idx = self.context_idx_bank[mask_sel]  # [B, 16]
            tgt_idx = self.target_idx_bank[mask_sel]   # [B, 9]
        else:
            ctx_idx = self.context_idx.unsqueeze(0).expand(B, -1)
            tgt_idx = self.target_idx.unsqueeze(0).expand(B, -1)

        ctx_tokens_a = tokens_a[batch_arange, ctx_idx]
        ctx_pos_a = pos_a[batch_arange, ctx_idx]
        tgt_tokens_b = tokens_b[batch_arange, tgt_idx]
        tgt_pos_b = pos_b[batch_arange, tgt_idx]

        # 4. Context encoder encodes visible context from Condition A
        H_ctx_a = self.context_encoder(ctx_tokens_a, ctx_pos_a)  # [B, 16, D]

        # 5. Target encoder encodes masked core from Condition B (stop-gradient)
        with torch.no_grad():
            H_tgt_b = self.target_encoder(tgt_tokens_b, tgt_pos_b)  # [B, 9, D]

        if getattr(self.config, "relative_target", False) and c_b is not None:
            target_signal = H_tgt_b - c_b
        else:
            target_signal = H_tgt_b

        n_ctx = H_ctx_a.shape[1]
        scan_id_expanded = None
        scan_id_b_expanded = None
        if scan_id is not None:
            scan_id_expanded = scan_id.unsqueeze(1).expand(-1, n_ctx).reshape(-1)
            _sid_b = scan_id_b if scan_id_b is not None else scan_id
            scan_id_b_expanded = _sid_b.unsqueeze(1).expand(-1, n_ctx).reshape(-1)

        # 6. Prediction: L2-normalize c_B so an unbounded projector cannot turn the
        #    condition token into a per-scan bias bank (AdaLN has no output norm).
        cond_in = F.normalize(c_b, p=2.0, dim=-1) if c_b is not None else None
        H_pred_b = self.predictor(H_ctx_a, tgt_pos_b, condition_token=cond_in)

        pred_loss_type = getattr(self.config, "pred_loss_type", "l1")
        if pred_loss_type == "normalized_l1":
            pred_loss = normalized_probe_l1(H_pred_b, target_signal)
        else:
            pred_loss = F.l1_loss(H_pred_b, target_signal)

        # 7. VICReg on BOTH probe-level [B*16, D] and patch-level [B, D].
        #    Probe-level alone can be satisfied by static positional offsets
        #    pos[0..15] inside each window, while the per-pixel representation
        #    (r, c) collapses to a constant. Patch-level averaging removes those
        #    offsets, so the encoder must keep true spatial variance across the
        #    C-scan plane.
        flat_ctx = H_ctx_a.reshape(B * n_ctx, -1)
        patch_ctx = self._pool_spatial_context(H_ctx_a, ctx_idx, h_anchor=None)  # [B, D]
        if self.config.var_weight > 0 or self.config.cov_weight > 0:
            _use_sc = getattr(self.config, "scan_centered_vicreg", True)
            vic_probe, var_probe, cov_probe = compute_vicreg_loss(
                flat_ctx,
                var_weight=self.config.var_weight,
                cov_weight=self.config.cov_weight,
                gamma=self.config.var_gamma,
                scan_id=scan_id_expanded if _use_sc else None,
                min_group_size=2 * n_ctx,  # require >= 2 patches of the same scan
            )
            vic_patch, var_patch, cov_patch = compute_vicreg_loss(
                patch_ctx,
                var_weight=self.config.var_weight,
                cov_weight=self.config.cov_weight,
                gamma=self.config.var_gamma,
                scan_id=scan_id if _use_sc else None,
                min_group_size=2,          # require >= 2 patches of the same scan
            )
            vicreg_loss = 0.5 * (vic_probe + vic_patch)
            var_loss = 0.5 * (var_probe + var_patch)
            cov_loss = 0.5 * (cov_probe + cov_patch)
        else:
            vicreg_loss = torch.tensor(0.0, device=device)
            var_loss = torch.tensor(0.0, device=device)
            cov_loss = torch.tensor(0.0, device=device)

        # 8. Background-gated cross-condition invariance L_inv
        inv_weight = getattr(self.config, "inv_weight", 0.0)
        eff_inv = inv_weight * inv_scale
        if eff_inv > 0:
            with torch.no_grad():
                H_ctx_b = self.target_encoder(
                    tokens_b[batch_arange, ctx_idx], pos_b[batch_arange, ctx_idx]
                )
            z_a = flat_ctx.float()
            z_b = H_ctx_b.reshape(B * H_ctx_b.shape[1], -1).float()
            if getattr(self.config, "scan_centered_inv", True) and scan_id_expanded is not None:
                z_a = center_by_group(z_a, scan_id_expanded, min_group_size=2 * n_ctx)
                z_b = center_by_group(z_b, scan_id_b_expanded, min_group_size=2 * n_ctx)
            if self.inv_head is not None:
                z_a = self.inv_head(z_a)

            keep, keep_frac = self._background_gate(
                x_a, ctx_idx, batch_arange, float(getattr(self.config, "inv_trim_q", 0.75))
            )
            if keep is not None:
                z_a = z_a[keep]
                z_b = z_b[keep]
            if getattr(self.config, "inv_cosine", True):
                # fp32 + eps=1e-6: eps=1e-8 is not representable in fp16 and made
                # the cosine blow up under AMP (exp71a non-finite loss).
                inv_loss = 1.0 - F.cosine_similarity(z_a.float(), z_b.float(), dim=-1, eps=1e-6).mean()
            else:
                inv_loss = F.mse_loss(z_a.float(), z_b.float())
        else:
            inv_loss = torch.tensor(0.0, device=device)
            keep_frac = 1.0

        # 9. Total loss: prediction + VICReg + (optionally capped) L_inv.
        #    FM22: the cap must NEVER touch VICReg. Capping aux at ratio*pred makes
        #    the variance gradient proportional to pred_loss; under a scale-invariant
        #    prediction loss, rank collapse lowers pred_loss, which shrinks the cap,
        #    which starves VICReg further (exp71: R_eff 1.6D-2.8D, val_auc 0.53,
        #    aux pinned at exactly 0.25*pred every epoch, exp70 with no cap: R_eff
        #    40-62D). VICReg is the only force against constant collapse, so it is
        #    always applied at full strength; only the optional L_inv is capped.
        inv_term = eff_inv * inv_loss
        anti_hijack_ratio = getattr(self.config, "anti_hijack_ratio", 0.0)
        if anti_hijack_ratio > 0 and pred_loss.detach() > 0:
            max_inv = anti_hijack_ratio * pred_loss.detach()
            if inv_term > max_inv:
                inv_term = inv_term * (max_inv / (inv_term + 1e-8))
        aux_loss = vicreg_loss + inv_term
        total_loss = pred_loss + aux_loss

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "aux_loss": aux_loss,
            "inv_loss": inv_loss,
            "inv_keep_frac": torch.tensor(keep_frac, device=device),
            "H_ctx_a": H_ctx_a,
            "H_tgt_b": H_tgt_b,
            "H_pred_b": H_pred_b,
            "target_signal": target_signal,
        }

    @torch.no_grad()
    def extract_features(
        self,
        x: torch.Tensor,
        bg_ref_energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extracts the unified downstream representation [h_center, mean(h_context)] [B, 2*D].

        Args:
            x: [B, 5, 5, C] patches of a single inspection scan.
            bg_ref_energy: optional per-sample sound-metal background energy for the
                global Delta E anchor (identical definition as in training).
        """
        tokens, pos = self.tokenizer(x, bg_ref_energy=bg_ref_energy)
        h_all = self.context_encoder(tokens, pos)  # [B, 25, D]
        h_center = h_all[:, 0, :]
        # Probe index 0 is the center; 1..24 are the r1/r2/r3 rings.
        ring_idx = torch.arange(1, self.tokenizer.num_tokens, device=x.device)
        h_context = self._pool_spatial_context(h_all[:, 1:, :], ring_idx, h_anchor=h_center)
        return torch.cat([h_center, h_context], dim=-1)
