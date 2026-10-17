"""
Cross-Condition Paired JEPA Architecture for PECT.

Core Principles:
- Inputs: Pair (x_A, x_B) at identical physical coordinates from two distinct measurement conditions.
- Condition Extraction: Pure self-supervised condition token c_B extracted from background sound-metal patch ref_B.
- Masking: Context from outer ring of Condition A (16 probes), Target from inner core of Condition B (9 probes).
- Objective: Predict H_tgt(B) from H_ctx(A) conditioned on c_B using pure L1 loss + Context VICReg.
- Representation: Unified frozen representation Z = [h_center, h_context] in R^{2D} for all downstream tasks.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.config import CrossConditionConfig
from ..data.topologies import get_contiguous_star_masks
from .tokenizer import DualDomainTokenizer
from .context_encoder import ContextEncoder
from .target_encoder import TargetEncoder
from .predictor import CrossConditionPredictor


def compute_vicreg_loss(
    z: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
    scan_id: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VICReg coordinate-wise variance hinge and covariance decorrelation penalty.
    If scan_id is provided, z is centered per-scan (z_centered = z - mean_{scan}(z))
    before computing variance and covariance. This strictly prevents between-scan drift
    from falsely satisfying the variance regularizer, forcing the model to amplify
    true within-scan flaw/defect variance!

    Vectorized via scatter (index_add_) over 0..S-1 remapped ids: a single pass
    instead of one Python-level masked pass per unique scan (~100 kernel-launch
    rounds per training step -> 3 kernels). Singletons keep z unchanged.
    """
    z_f = z.float()
    if scan_id is not None:
        # z: [N, D], scan_id: [N] (arbitrary ids)
        uniq, inv = torch.unique(scan_id, return_inverse=True)  # inv: [N] in 0..S-1
        n_groups = uniq.numel()
        # float32 accumulators for stable means and covariance under AMP fp16 inputs
        counts = torch.bincount(inv, minlength=n_groups)  # [S] int64
        sums = torch.zeros(n_groups, z_f.shape[1], device=z_f.device, dtype=torch.float32)
        sums.index_add_(0, inv, z_f)
        means = sums / counts.clamp_min(1).unsqueeze(1).float()  # [S, D]
        m = means[inv]  # [N, D] float32
        singleton = (counts[inv] <= 1).unsqueeze(1)  # [N, 1]
        z_cent = torch.where(singleton, z_f, z_f - m)
        # Standard deviation computed strictly on within-scan centered perturbations
        std = torch.sqrt(z_cent.var(dim=0, unbiased=False) + eps)
    else:
        std = torch.sqrt(z_f.var(dim=0, unbiased=False) + eps)
        z_cent = z_f - z_f.mean(dim=0, keepdim=True)

    var_loss = torch.mean(F.relu(gamma - std))

    N = z_cent.shape[0]
    cov = (z_cent.T @ z_cent) / max(N - 1, 1)
    cov_off = cov - torch.diag(torch.diag(cov))
    cov_loss = (cov_off ** 2).sum() / z_f.shape[1]

    total = var_weight * var_loss + cov_weight * cov_loss
    return total, var_loss, cov_loss


class CrossConditionJEPA(nn.Module):
    """
    Self-Supervised Cross-Condition Paired JEPA Model (Zero Metadata Injection).
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config

        # 1. Physical Dual-Domain Tokenizer
        self.tokenizer = DualDomainTokenizer(config)

        # 2. Context Encoder (Condition A)
        self.context_encoder = ContextEncoder(config)

        # 3. Target Encoder (Condition B, EMA teacher)
        self.target_encoder = TargetEncoder(config)
        self._init_target_encoder()

        # 4. Self-Supervised Condition Projector
        self.cond_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # 5. Cross-Condition Predictor
        self.predictor = CrossConditionPredictor(config)

        # 6. z_c grounding head: gives the carrier subspace an explicit job
        # (predict Condition-B token from pooled z_c) so the [z_c, z_s]
        # split is enforced by an objective instead of being decorative.
        # Inactive unless latent_disentanglement and zc_cond_weight > 0.
        if getattr(config, "latent_disentanglement", False):
            _D_c = getattr(config, "carrier_dim", config.embed_dim // 2)
            self.zc_cond_head = nn.Sequential(
                nn.Linear(_D_c, _D_c),
                nn.GELU(),
                nn.Linear(_D_c, config.embed_dim),
            )
        else:
            self.zc_cond_head = None

        # 7. Asymmetric-inv predictor head (BYOL-style): absorbs the
        # cross-condition alignment work so the z_s encoder itself is never
        # directly pulled toward the target branch. Created ONLY when both
        # disentanglement and inv_predictor are on: legacy configs get no new
        # parameters, so old checkpoints keep loading with strict=True.
        if getattr(config, "latent_disentanglement", False) and getattr(config, "inv_predictor", False):
            _D_s = getattr(config, "scattering_dim", config.embed_dim - getattr(config, "carrier_dim", config.embed_dim // 2))
            _H = max(2 * _D_s, 64)
            self.inv_head = nn.Sequential(
                nn.Linear(_D_s, _H),
                nn.GELU(),
                nn.Linear(_H, _D_s),
            )
        else:
            self.inv_head = None

        # Concentric star partitioning:
        # Indices 0..8: Core 9 probes (center + 8 inner ring probes)
        # Indices 9..24: Outer 16 probes (outer rings)
        self.register_buffer("target_idx", torch.arange(config.num_target_probes, dtype=torch.long))
        self.register_buffer("context_idx", torch.arange(config.num_target_probes, 25, dtype=torch.long))

        # Contiguous non-enclosing wedge mask bank (16 masks: 8 Family A outer half-plane wedges
        # with Center 0 in Context + 8 Family B apex-inclusive wedges with Center 0 in Target).
        # Registered with persistent=False so legacy checkpoints load with strict=True.
        tgt_bank_np, ctx_bank_np = get_contiguous_star_masks(
            star_radii=tuple(getattr(config, "star_radii", (1, 3, 7)))
        )
        self.register_buffer("target_idx_bank", torch.from_numpy(tgt_bank_np).long(), persistent=False)
        self.register_buffer("context_idx_bank", torch.from_numpy(ctx_bank_np).long(), persistent=False)

    def _init_target_encoder(self):
        """Initialize Target Encoder weights identical to Context Encoder."""
        for p_tgt, p_ctx in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad = False

    def update_target_encoder(self, momentum: float = 0.996):
        """Update Target Encoder parameters with EMA momentum."""
        self.target_encoder.update_ema(self.context_encoder, momentum=momentum)

    def extract_condition_token(self, ref_b: torch.Tensor) -> torch.Tensor:
        """
        Extracts condition token c_B from sound-metal background patch ref_b of Condition B.
        Args:
            ref_b: [B, 5, 5, C] raw background patch.
        Returns:
            c_b: [B, 1, D] transfer function condition token.
        """
        with torch.no_grad():
            ref_tokens, _ = self.tokenizer(ref_b)  # [B, 25, D]
            ref_mean = torch.mean(ref_tokens, dim=1)  # [B, D]
        return self.cond_proj(ref_mean).unsqueeze(1)  # [B, 1, D]

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        ref_b: torch.Tensor,
        scan_id: Optional[torch.Tensor] = None,
        inv_scale: float = 1.0,
        xcov_scale: float = 1.0,
        inv_sound_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass during self-supervised pretraining:
        x_a:     [B, 5, 5, C] measurement patch from Condition A
        x_b:     [B, 5, 5, C] measurement patch from Condition B (same physical coordinate)
        ref_b:   [B, 5, 5, C] background sound-metal patch of Condition B
        scan_id: [B] scan pair identifier for within-scan centered regularization
        inv_scale/xcov_scale: P0-1 curriculum multipliers in [0, 1] supplied by
            the trainer from the epoch schedule (default 1.0 = legacy path).
        """
        B = x_a.shape[0]
        device = x_a.device

        # 1. Dual-domain tokenization
        tokens_a, pos_a = self.tokenizer(x_a)  # [B, 25, D]
        tokens_b, pos_b = self.tokenizer(x_b)  # [B, 25, D]

        # 2. Extract self-supervised condition token (disabled if use_condition_token is False)
        c_b = self.extract_condition_token(ref_b) if self.config.use_condition_token else None

        # 3. Partition tokens:
        # - If contiguous_masking=True: sample from 16 spatially contiguous, non-enclosing
        #   wedge partitions (neither context inside target nor target surrounded by context),
        #   ensuring all 25 probes (including Center 0) are trained in ContextEncoder.
        # - Else (legacy default): fixed 16 outer context (9..24) vs 9 inner target (0..8).
        batch_arange = torch.arange(B, device=device).unsqueeze(1)
        if getattr(self.config, "contiguous_masking", False):
            num_masks = self.target_idx_bank.shape[0]
            if self.training:
                mask_sel = torch.randint(0, num_masks, (B,), device=device)
            else:
                mask_sel = torch.arange(B, device=device) % num_masks
            ctx_idx = self.context_idx_bank[mask_sel]  # [B, 16]
            tgt_idx = self.target_idx_bank[mask_sel]   # [B, 9]
        else:
            ctx_idx = self.context_idx.unsqueeze(0).expand(B, -1)  # [B, 16]
            tgt_idx = self.target_idx.unsqueeze(0).expand(B, -1)   # [B, 9]

        ctx_tokens_a = tokens_a[batch_arange, ctx_idx]  # [B, 16, D]
        ctx_pos_a = pos_a[batch_arange, ctx_idx]

        tgt_tokens_b = tokens_b[batch_arange, tgt_idx]  # [B, 9, D]
        tgt_pos_b = pos_b[batch_arange, tgt_idx]

        # 4. Context Encoder encodes visible context from Condition A
        H_ctx_a = self.context_encoder(ctx_tokens_a, ctx_pos_a)  # [B, 16, D]

        # 5. Target Encoder encodes masked core from Condition B (stop-gradient)
        with torch.no_grad():
            H_tgt_b = self.target_encoder(tgt_tokens_b, tgt_pos_b)  # [B, 9, D]

        # Target relative perturbation: Delta H_tgt = H_tgt - c_b
        use_relative = getattr(self.config, "relative_target", True)
        if use_relative and c_b is not None:
            target_signal = H_tgt_b - c_b  # [B, 9, D] centered around Condition B sound metal
        else:
            target_signal = H_tgt_b

        # Expand scan_id across the N_ctx context tokens if provided: [B] -> [B * N_ctx]
        # (N_ctx is read from the tensor, not hardcoded: masking splits vary)
        scan_id_expanded = None
        if scan_id is not None and getattr(self.config, "scan_centered_vicreg", True):
            scan_id_expanded = scan_id.unsqueeze(1).expand(-1, H_ctx_a.shape[1]).reshape(-1)

        # 6. Latent Disentanglement vs Standard Prediction
        if getattr(self.config, "latent_disentanglement", False):
            # Split Context and Target into Carrier z_c and Scattering z_s
            # D = D_c + D_s
            D_c = getattr(self.config, "carrier_dim", self.config.embed_dim // 2)
            D_s = getattr(self.config, "scattering_dim", self.config.embed_dim - D_c)

            H_ctx_c = H_ctx_a[..., :D_c]  # [B, 16, D_c] Carrier background
            H_ctx_s = H_ctx_a[..., D_c:]  # [B, 16, D_s] Scattering flaw perturbation

            H_tgt_c = target_signal[..., :D_c]  # [B, 9, D_c]
            H_tgt_s = target_signal[..., D_c:]  # [B, 9, D_s]

            # A) Scattering Prediction: Predict z_s WITHOUT condition token (unconditioned mapping)
            # If predictor_scattering_only=True, zero out H_ctx_c before predictor so predictor
            # cannot use Condition A's carrier subspace as a domain-translation shortcut.
            if getattr(self.config, "predictor_scattering_only", False):
                H_ctx_for_pred = torch.cat([torch.zeros_like(H_ctx_c), H_ctx_s], dim=-1)
            else:
                H_ctx_for_pred = H_ctx_a
            H_pred_b = self.predictor(H_ctx_for_pred, tgt_pos_b, condition_token=None)  # [B, 9, D]
            H_pred_s = H_pred_b[..., D_c:]
            pred_loss = F.l1_loss(H_pred_s, H_tgt_s)

            # B) Spatial Flatness on Carrier z_c: In a local 5x5 patch and across the scan, z_c should be spatially flat
            # Penalize variance across the 16 context probes for each patch
            flat_carrier_loss = torch.mean(torch.var(H_ctx_c, dim=1, unbiased=False))
            carrier_loss = self.config.carrier_flat_weight * flat_carrier_loss

            # C) VICReg on Scattering z_s: Preserve multi-dimensional rank and prevent 1D collapse
            flat_ctx_s = H_ctx_s.reshape(B * H_ctx_s.shape[1], -1)
            vicreg_loss, var_loss, cov_loss = compute_vicreg_loss(
                flat_ctx_s,
                var_weight=self.config.scattering_var_weight,
                cov_weight=self.config.scattering_cov_weight,
                gamma=self.config.var_gamma,
                scan_id=scan_id_expanded,
            )

            # C2) Fix 4: var-hinge on the POOLED scattering vector itself
            # ([center, mean-rest] = the actual downstream readout quantity).
            # Token-level VICReg above does not protect the pooled features,
            # where needle collapse still happens. scan_id used per-patch
            # (NOT token-expanded): correct grouping at this granularity.
            pool_var_weight = getattr(self.config, "pool_var_weight", 0.0)
            if pool_var_weight > 0:
                H_pool_s = torch.cat([H_ctx_s[:, 0, :], H_ctx_s[:, 1:, :].mean(dim=1)], dim=-1)  # [B, 2*D_s]
                pool_var_loss, _, _ = compute_vicreg_loss(
                    H_pool_s,
                    var_weight=1.0,
                    cov_weight=0.0,
                    gamma=self.config.var_gamma,
                    scan_id=scan_id,
                )
            else:
                pool_var_loss = torch.tensor(0.0, device=device)

            # D) EXP-49 L_inv: Cross-condition invariance on z_s.
            # Same physical coordinate seen under condition A (context encoder,
            # grad ON) vs condition B (target encoder, stop-grad/EMA target).
            # Penalizes any condition drift left inside z_s.
            inv_weight = getattr(self.config, "inv_weight", 0.0)
            eff_inv = inv_weight * inv_scale
            if eff_inv > 0:
                # z_s of condition B from the EMA target encoder over the SAME
                # context probes (context geometry matches H_ctx_a).
                with torch.no_grad():
                    ctx_tokens_b = tokens_b[batch_arange, ctx_idx]  # [B, 16, D]
                    ctx_pos_b = pos_b[batch_arange, ctx_idx]
                    H_ctx_b = self.target_encoder(ctx_tokens_b, ctx_pos_b)  # [B, 16, D]
                H_ctx_b_s = H_ctx_b[..., D_c:]  # [B, N_ctx, D_s] fixed target
                H_inv_online = self.inv_head(H_ctx_s) if self.inv_head is not None else H_ctx_s

                # If scan_centered_inv=True, remove batch/scan DC offset before cosine L_inv
                # so LayerNorm bias cannot trivially satisfy 1 - cos(mu + dA, mu + dB) -> 0.
                if getattr(self.config, "scan_centered_inv", False):
                    H_inv_online = H_inv_online - H_inv_online.mean(dim=(0, 1), keepdim=True)
                    H_ctx_b_s = H_ctx_b_s - H_ctx_b_s.mean(dim=(0, 1), keepdim=True)

                use_oracle = bool(getattr(self.config, "inv_oracle_mask", False))
                snd_frac = torch.tensor(1.0, device=device)
                if use_oracle and inv_sound_mask is not None:
                    if inv_sound_mask.ndim == 2 and inv_sound_mask.shape[1] == 25:
                        m_selected = inv_sound_mask[batch_arange, ctx_idx]
                    elif getattr(self.config, "contiguous_masking", False):
                        raise ValueError(
                            f"inv_oracle_mask=True with contiguous_masking=True requires "
                            f"inv_sound_mask of shape [B, 25], got {tuple(inv_sound_mask.shape)}."
                        )
                    else:
                        m_selected = inv_sound_mask
                    m = m_selected.reshape(-1).to(dtype=torch.bool)
                    frac = m.float().mean()
                    snd_frac = frac.detach()
                    if m.sum() >= max(1, int(0.05 * m.numel())):
                        zo = H_inv_online.reshape(-1, H_inv_online.shape[-1])[m]
                        zt = H_ctx_b_s.reshape(-1, H_ctx_b_s.shape[-1])[m]
                        if getattr(self.config, "inv_cosine", False):
                            inv_loss = 1.0 - F.cosine_similarity(zo, zt, dim=-1).mean()
                        else:
                            inv_loss = F.mse_loss(zo, zt)
                    else:
                        if getattr(self.config, "inv_cosine", False):
                            inv_loss = 1.0 - F.cosine_similarity(H_inv_online, H_ctx_b_s, dim=-1).mean()
                        else:
                            inv_loss = F.mse_loss(H_inv_online, H_ctx_b_s)
                elif getattr(self.config, "inv_cosine", False):
                    inv_loss = 1.0 - F.cosine_similarity(H_inv_online, H_ctx_b_s, dim=-1).mean()
                else:
                    inv_loss = F.mse_loss(H_inv_online, H_ctx_b_s)
                inv_sound_frac = snd_frac
            else:
                # Skipped by schedule: avoid the extra EMA forward entirely.
                inv_loss = torch.tensor(0.0, device=device)
                inv_sound_frac = torch.tensor(1.0, device=device)

            # E) EXP-49 L_xcov: Cross-subspace decorrelation lock.
            # ||XCov(z_c, z_s)||^2 over the N = B*16 context tokens.
            # Purely unsupervised; prevents z_c from copying defect info back
            # from z_s, forcing it to pick up the remainder (condition drift).
            # NOTE: this term alone cannot create the split (it is symmetric);
            # L_inv above creates the role asymmetry, L_xcov locks it in.
            xcov_weight = getattr(self.config, "xcov_weight", 0.0)
            eff_xcov = xcov_weight * xcov_scale
            if eff_xcov > 0:
                flat_ctx_c = H_ctx_c.reshape(B * H_ctx_c.shape[1], -1).float()  # [N, D_c]
                flat_ctx_s_f = flat_ctx_s.float()
                N = flat_ctx_c.shape[0]
                zc_cent = flat_ctx_c - flat_ctx_c.mean(dim=0, keepdim=True)
                zs_cent = flat_ctx_s_f - flat_ctx_s_f.mean(dim=0, keepdim=True)
                xcov = (zc_cent.T @ zs_cent) / max(N - 1, 1)  # [D_c, D_s]
                xcov_loss = (xcov ** 2).sum() / (D_c * D_s)
            else:
                xcov_loss = torch.tensor(0.0, device=device)

            # F) z_c grounding job: predict the Condition-B token from pooled
            # z_c. Gives the carrier subspace an explicit condition-carrying
            # role (target detached: gradient shapes z_c + head, never the
            # condition path). Inactive when weight is 0 or c_b is disabled.
            zc_cond_weight = getattr(self.config, "zc_cond_weight", 0.0)
            if zc_cond_weight > 0 and self.zc_cond_head is not None and c_b is not None:
                zc_pred = self.zc_cond_head(H_ctx_c.mean(dim=1))  # [B, D]
                zc_cond_loss = F.mse_loss(zc_pred, c_b.squeeze(1).detach())
            else:
                zc_cond_loss = torch.tensor(0.0, device=device)

            # G) Fix 1: kill the shared DC offset (needle geometry). The offset
            # is in the null space of every other term (cancels in branch
            # differences; VICReg/xcov center first), so it random-walks via
            # LN gain/bias and makes all cosine reasoning vacuous. Penalize
            # GLOBAL batch-token mean-square per subspace: per-scan condition
            # bias in z_c is legitimate and survives, only the shared giant
            # bias is removed.
            offset_weight = getattr(self.config, "offset_weight", 0.0)
            if offset_weight > 0:
                m_c = H_ctx_c.reshape(-1, H_ctx_c.shape[-1]).mean(dim=0)
                m_s = H_ctx_s.reshape(-1, H_ctx_s.shape[-1]).mean(dim=0)
                offset_loss = (m_c.pow(2).mean() + m_s.pow(2).mean())
            else:
                offset_loss = torch.tensor(0.0, device=device)

            total_loss = (
                pred_loss
                + carrier_loss
                + vicreg_loss
                + eff_inv * inv_loss
                + eff_xcov * xcov_loss
                + zc_cond_weight * zc_cond_loss
                + offset_weight * offset_loss
                + pool_var_weight * pool_var_loss
            )
        else:
            # Standard single-space prediction
            H_pred_b = self.predictor(H_ctx_a, tgt_pos_b, c_b)  # [B, 9, D]
            pred_loss = F.l1_loss(H_pred_b, target_signal)

            flat_ctx = H_ctx_a.reshape(B * H_ctx_a.shape[1], -1)
            if self.config.var_weight > 0 or self.config.cov_weight > 0:
                vicreg_loss, var_loss, cov_loss = compute_vicreg_loss(
                    flat_ctx,
                    var_weight=self.config.var_weight,
                    cov_weight=self.config.cov_weight,
                    gamma=self.config.var_gamma,
                    scan_id=scan_id_expanded,
                )
            else:
                vicreg_loss = torch.tensor(0.0, device=device)
                var_loss = torch.tensor(0.0, device=device)
                cov_loss = torch.tensor(0.0, device=device)

            carrier_loss = torch.tensor(0.0, device=device)
            total_loss = pred_loss + vicreg_loss
            inv_loss = torch.tensor(0.0, device=device)
            inv_sound_frac = torch.tensor(1.0, device=device)
            xcov_loss = torch.tensor(0.0, device=device)
            zc_cond_loss = torch.tensor(0.0, device=device)
            offset_loss = torch.tensor(0.0, device=device)

            # Fix 4 (standard branch): pooled hinge on the readout quantity.
            pool_var_weight = getattr(self.config, "pool_var_weight", 0.0)
            if pool_var_weight > 0:
                H_pool = torch.cat([H_ctx_a[:, 0, :], H_ctx_a[:, 1:, :].mean(dim=1)], dim=-1)
                pool_var_loss, _, _ = compute_vicreg_loss(
                    H_pool,
                    var_weight=1.0,
                    cov_weight=0.0,
                    gamma=self.config.var_gamma,
                    scan_id=scan_id,
                )
                total_loss = total_loss + pool_var_weight * pool_var_loss
            else:
                pool_var_loss = torch.tensor(0.0, device=device)

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "carrier_loss": carrier_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "inv_loss": inv_loss,
            "inv_sound_frac": inv_sound_frac,
            "xcov_loss": xcov_loss,
            "zc_cond_loss": zc_cond_loss,
            "offset_loss": offset_loss,
            "pool_var_loss": pool_var_loss,
            "H_ctx_a": H_ctx_a,
            "H_tgt_b": H_tgt_b,
            "H_pred_b": H_pred_b,
            "target_signal": target_signal,
        }

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts unified downstream representation for inspection C-scan:
        If latent_disentanglement is True:
          - 'scattering': Extracts scattering perturbation z_s = [h_center_s, h_context_s] in R^{2*D_s} (drift-free).
          - 'carrier': Extracts carrier baseline z_c in R^{2*D_c}.
          - 'full': Extracts full concatenated vector [z_c, z_s] in R^{2*D}.
        Default: Concatenated [B, 2*D] vector [h_center, h_context_pool].
        """
        tokens, pos = self.tokenizer(x)  # [B, 25, D]
        h_all = self.context_encoder(tokens, pos)  # [B, 25, D]

        if getattr(self.config, "latent_disentanglement", False):
            D_c = getattr(self.config, "carrier_dim", self.config.embed_dim // 2)
            latent_type = getattr(self.config, "downstream_latent_type", "scattering")

            if latent_type == "scattering":
                h_all_s = h_all[..., D_c:]  # [B, 25, D_s]
                h_center = h_all_s[:, 0, :]
                h_context = torch.mean(h_all_s[:, 1:, :], dim=1)
                return torch.cat([h_center, h_context], dim=-1)  # [B, 2*D_s]
            elif latent_type == "carrier":
                h_all_c = h_all[..., :D_c]  # [B, 25, D_c]
                h_center = h_all_c[:, 0, :]
                h_context = torch.mean(h_all_c[:, 1:, :], dim=1)
                return torch.cat([h_center, h_context], dim=-1)  # [B, 2*D_c]

        h_center = h_all[:, 0, :]  # [B, D] Center probe
        h_context = torch.mean(h_all[:, 1:, :], dim=1)  # [B, D] Outer context pool
        return torch.cat([h_center, h_context], dim=-1)  # [B, 2*D]
