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
    if scan_id is not None:
        # z: [N, D], scan_id: [N] (arbitrary ids)
        uniq, inv = torch.unique(scan_id, return_inverse=True)  # inv: [N] in 0..S-1
        n_groups = uniq.numel()
        # float32 accumulators for stable means under AMP fp16 inputs
        counts = torch.bincount(inv, minlength=n_groups)  # [S] int64
        sums = torch.zeros(n_groups, z.shape[1], device=z.device, dtype=torch.float32)
        sums.index_add_(0, inv, z.float())
        means = sums / counts.clamp_min(1).unsqueeze(1).float()  # [S, D]
        m = means[inv].to(z.dtype)  # [N, D]
        singleton = (counts[inv] <= 1).unsqueeze(1)  # [N, 1]
        z_cent = torch.where(singleton, z, z - m)
        # Standard deviation computed strictly on within-scan centered perturbations
        std = torch.sqrt(z_cent.var(dim=0, unbiased=False) + eps)
    else:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        z_cent = z - z.mean(dim=0, keepdim=True)

    var_loss = torch.mean(F.relu(gamma - std))

    N = z_cent.shape[0]
    cov = (z_cent.T @ z_cent) / max(N - 1, 1)
    cov_off = cov - torch.diag(torch.diag(cov))
    cov_loss = (cov_off ** 2).sum() / z.shape[1]

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

        # Concentric star partitioning:
        # Indices 0..8: Core 9 probes (center + 8 inner ring probes)
        # Indices 9..24: Outer 16 probes (outer rings)
        self.register_buffer("target_idx", torch.arange(config.num_target_probes, dtype=torch.long))
        self.register_buffer("context_idx", torch.arange(config.num_target_probes, 25, dtype=torch.long))

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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass during self-supervised pretraining:
        x_a:     [B, 5, 5, C] measurement patch from Condition A
        x_b:     [B, 5, 5, C] measurement patch from Condition B (same physical coordinate)
        ref_b:   [B, 5, 5, C] background sound-metal patch of Condition B
        scan_id: [B] scan pair identifier for within-scan centered regularization
        """
        B = x_a.shape[0]
        device = x_a.device

        # 1. Dual-domain tokenization
        tokens_a, pos_a = self.tokenizer(x_a)  # [B, 25, D]
        tokens_b, pos_b = self.tokenizer(x_b)  # [B, 25, D]

        # 2. Extract self-supervised condition token (disabled if use_condition_token is False)
        c_b = self.extract_condition_token(ref_b) if self.config.use_condition_token else None

        # 3. Partition tokens: Context from A (16 outer), Target from B (9 inner)
        batch_arange = torch.arange(B, device=device).unsqueeze(1)
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

        # Expand scan_id across the 16 context tokens if provided: [B] -> [B * 16]
        scan_id_expanded = None
        if scan_id is not None and getattr(self.config, "scan_centered_vicreg", True):
            scan_id_expanded = scan_id.unsqueeze(1).expand(-1, 16).reshape(-1)

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
            # This forces z_s to align sign/polarity across sensors (Hall vs TMR)
            H_pred_b = self.predictor(H_ctx_a, tgt_pos_b, condition_token=None)  # [B, 9, D]
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

            # D) EXP-49 L_inv: Cross-condition invariance on z_s.
            # Same physical coordinate seen under condition A (context encoder,
            # grad ON) vs condition B (target encoder, stop-grad/EMA target).
            # Penalizes any condition drift left inside z_s. Asymmetric by
            # design: gradient flows only into the A branch, the B branch is
            # a fixed per-step target (BYOL-style, no collapse).
            inv_weight = getattr(self.config, "inv_weight", 0.0)
            if inv_weight > 0:
                # z_s of condition B from the EMA target encoder over the SAME
                # 16 outer context probes (context geometry matches H_ctx_a).
                with torch.no_grad():
                    ctx_tokens_b = tokens_b[batch_arange, ctx_idx]  # [B, 16, D]
                    ctx_pos_b = pos_b[batch_arange, ctx_idx]
                    H_ctx_b = self.target_encoder(ctx_tokens_b, ctx_pos_b)  # [B, 16, D]
                H_ctx_b_s = H_ctx_b[..., D_c:]  # [B, 16, D_s] fixed target
                inv_loss = F.mse_loss(H_ctx_s, H_ctx_b_s)
            else:
                inv_loss = torch.tensor(0.0, device=device)

            # E) EXP-49 L_xcov: Cross-subspace decorrelation lock.
            # ||XCov(z_c, z_s)||^2 over the N = B*16 context tokens.
            # Purely unsupervised; prevents z_c from copying defect info back
            # from z_s, forcing it to pick up the remainder (condition drift).
            # NOTE: this term alone cannot create the split (it is symmetric);
            # L_inv above creates the role asymmetry, L_xcov locks it in.
            xcov_weight = getattr(self.config, "xcov_weight", 0.0)
            if xcov_weight > 0:
                flat_ctx_c = H_ctx_c.reshape(B * H_ctx_c.shape[1], -1)  # [N, D_c]
                N = flat_ctx_c.shape[0]
                zc_cent = flat_ctx_c - flat_ctx_c.mean(dim=0, keepdim=True)
                zs_cent = flat_ctx_s - flat_ctx_s.mean(dim=0, keepdim=True)
                xcov = (zc_cent.T @ zs_cent) / max(N - 1, 1)  # [D_c, D_s]
                xcov_loss = (xcov ** 2).sum() / (D_c * D_s)
            else:
                xcov_loss = torch.tensor(0.0, device=device)

            total_loss = (
                pred_loss
                + carrier_loss
                + vicreg_loss
                + inv_weight * inv_loss
                + xcov_weight * xcov_loss
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
            xcov_loss = torch.tensor(0.0, device=device)

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "carrier_loss": carrier_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "inv_loss": inv_loss,
            "xcov_loss": xcov_loss,
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
