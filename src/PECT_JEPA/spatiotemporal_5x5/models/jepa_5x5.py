"""
Unified 5x5 Spatiotemporal PECT-JEPA Model.
Integrates the SpatialGridTokenizer5x5, ContiguousClusterMasker5x5,
ContextEncoder5x5, TargetEncoder5x5, Predictor5x5, and JEPALoss5x5.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.config import Spatiotemporal5x5Config, get_default_config_5x5
from .tokenizer_5x5 import (
    DualScaleDiffusionTokenizer5x5,
    SpatialGridTokenizer5x5,
    DualDomainGridTokenizer5x5,
    DualDomainAttentionTokenizer5x5,
    build_tokenizer_5x5,
)
from .context_encoder import ContextEncoder5x5
from .target_encoder import TargetEncoder5x5
from .predictor import Predictor5x5, OperatorDiffusionPredictor5x5, build_predictor_5x5
from ..masking.cluster_mask import (
    build_masker_5x5,
    ContiguousClusterMasker5x5,
    SpatiotemporalDiffusionMasker5x5,
)
from ..losses.jepa_loss import JEPALoss5x5


class PECT_JEPA_5x5(nn.Module):
    """
    Unified 5x5 Spatiotemporal PECT-JEPA Self-Supervised Model.
    Input: [B, 5, 5, C] local C-scan grid (25 points, C channels).
    """

    def __init__(self, config: Optional[Spatiotemporal5x5Config] = None):
        super().__init__()
        if config is None:
            config = get_default_config_5x5()
        self.config = config

        # 1. Tokenizer (Dual-Scale Diffusion, Dual-Domain Attention, or Time-Only)
        self.tokenizer = build_tokenizer_5x5(config)

        # 2. Masker (Spatiotemporal Diffusion 3D or Contiguous Cluster 2D)
        self.masker = build_masker_5x5(config)

        # 3. Context Encoder
        self.context_encoder = ContextEncoder5x5(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            use_radial_attention_bias=getattr(config, "use_radial_attention_bias", False),
            spatial_topology=getattr(config, "spatial_topology", "concentric_star"),
            star_radii=getattr(config, "star_radii", (1, 3, 7)),
            grid_size=config.grid_size,
            diffusion_gamma_init=getattr(config, "diffusion_gamma_init", 0.05),
            diffusion_alpha_init=getattr(config, "diffusion_alpha_init", 0.1),
        )

        # 4. Target Encoder (EMA)
        self.target_encoder = TargetEncoder5x5(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            use_radial_attention_bias=getattr(config, "use_radial_attention_bias", False),
            spatial_topology=getattr(config, "spatial_topology", "concentric_star"),
            star_radii=getattr(config, "star_radii", (1, 3, 7)),
            grid_size=config.grid_size,
            diffusion_gamma_init=getattr(config, "diffusion_gamma_init", 0.05),
            diffusion_alpha_init=getattr(config, "diffusion_alpha_init", 0.1),
        )
        self._init_target_encoder()

        # 5. Predictor (Physics Operator Diffusion Predictor or Standard)
        self.predictor = build_predictor_5x5(config)

        # 6. Loss Function with VICReg Coordinate-Wise Regularization & Physics Alignments
        self.loss_fn = JEPALoss5x5(
            loss_type=config.loss_type,
            eps=config.eps,
            liftoff_invar_weight=getattr(config, "liftoff_invar_weight", 0.0),
            phase_align_weight=getattr(config, "phase_align_weight", 0.0),
            fluct_weight=getattr(config, "fluct_weight", 2.0),
            adaptive_disturbance_weight=getattr(config, "adaptive_disturbance_weight", 2.0),
            temporal_mono_weight=getattr(config, "temporal_mono_weight", 0.05),
            var_weight=getattr(config, "var_weight", 1.0),
            cov_weight=getattr(config, "cov_weight", 1.0),
            var_gamma=getattr(config, "var_gamma", 1.0),
            uniformity_weight=getattr(config, "uniformity_weight", 0.0),
            uniformity_t=getattr(config, "uniformity_t", 2.0),
            uniformity_subsample=getattr(config, "uniformity_subsample", 1024),
            norm_floor_weight=getattr(config, "norm_floor_weight", 0.0),
            norm_floor_target=getattr(config, "norm_floor_target", 1.0),
        )

        # 7. Physical Alignment Modules (Option B)
        self.depth_head = nn.Linear(config.embed_dim, 1, bias=False)
        nn.init.trunc_normal_(self.depth_head.weight, std=0.02)

    @staticmethod
    def compute_characteristic_frequency(x: torch.Tensor, num_bins: int = 14) -> torch.Tensor:
        """
        Computes the normalized Energy-Weighted Characteristic Frequency (omega_bar)
        for each sample in the batch:
            omega_bar = sum_{k=1}^K omega_k * |X_k|^2 / sum_{k=1}^K |X_k|^2
        Returns:
            freq_bar: [B] in range (0, 1]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C).float()
        # Compute FFT along temporal dimension
        X_fft = torch.fft.rfft(x_flat, dim=-1)[:, :, 1:num_bins + 1]  # [B, 25, K], exclude DC
        pwr = torch.sum(torch.abs(X_fft) ** 2, dim=1)  # [B, K], spatial average power per freq bin
        # Frequency bins: 1, 2, ..., K normalized to (0, 1]
        k_indices = torch.linspace(1.0 / float(num_bins), 1.0, num_bins, device=x.device, dtype=torch.float32)  # [K]
        total_pwr = torch.sum(pwr, dim=-1, keepdim=True) + 1e-12
        weights = pwr / total_pwr  # [B, K]
        freq_bar = torch.sum(weights * k_indices.unsqueeze(0), dim=-1)  # [B]
        return freq_bar.clamp(min=1.0 / float(num_bins), max=1.0)

    def _init_target_encoder(self):
        """Copy initial weights from Context Encoder to Target Encoder."""
        for p_tgt, p_ctx in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad = False

    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update Target Encoder weights via EMA (no-op if use_target_ema=False)."""
        if not getattr(self.config, "use_target_ema", False):
            return
        if momentum is None:
            momentum = self.config.ema_momentum
        self.target_encoder.update_ema(self.context_encoder, momentum=momentum)

    def forward(
        self,
        x: torch.Tensor,
        custom_context_indices: Optional[torch.Tensor] = None,
        custom_target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward self-supervised step with Physics Operator Diffusion & Physical Alignment.
        Args:
            x: [B, 5, 5, C] input grid
            custom_context_indices: optional override [B, N_ctx]
            custom_target_indices:  optional override [B, N_tgt]
            freq_condition: optional explicit frequency diffusion query [B]

        Returns dict of loss and representations.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        device = x.device

        # 1. Tokenization -> tokens [B, N_total, D], pos [B, N_total, D] (N_total is 50 for dual_scale or 25)
        tokens, pos = self.tokenizer(x)
        N_total = tokens.shape[1]

        # 2. Sample or use custom mask
        if custom_context_indices is not None and custom_target_indices is not None:
            context_indices = custom_context_indices.to(device)
            target_indices = custom_target_indices.to(device)
            mask_bool = torch.zeros(B, N_total, dtype=torch.bool, device=device)
            mask_bool.scatter_(1, target_indices, True)
        else:
            context_indices, target_indices, mask_bool = self.masker.sample_mask(B, device=device)

        # 3. Partition tokens
        batch_arange = torch.arange(B, device=device).unsqueeze(1)
        context_tokens = tokens[batch_arange, context_indices]
        context_pos = pos[batch_arange, context_indices]
        target_tokens = tokens[batch_arange, target_indices]
        target_pos = pos[batch_arange, target_indices]

        # 4. Context Encoder (only sees visible context tokens)
        H_ctx = self.context_encoder(context_tokens, context_pos, context_indices=context_indices)

        # 5. Predictor (Predicts target representation from context and target queries)
        is_physics_predictor = hasattr(self.predictor, "op_embedding")
        delta_pred = None
        if hasattr(self.predictor, "residual_head"):
            if freq_condition is None:
                freq_condition = self.compute_characteristic_frequency(
                    x, num_bins=getattr(self.config, "num_freq_bins", 14)
                )
            H_pred, delta_pred, _ = self.predictor(
                H_context=H_ctx,
                target_pos=target_pos,
                context_indices=context_indices,
                target_indices=target_indices,
                freq_condition=freq_condition,
                return_residual=True,
            )
        elif is_physics_predictor:
            if freq_condition is None:
                freq_condition = self.compute_characteristic_frequency(
                    x, num_bins=getattr(self.config, "num_freq_bins", 14)
                )
            H_pred = self.predictor(
                H_context=H_ctx,
                target_pos=target_pos,
                context_indices=context_indices,
                target_indices=target_indices,
                freq_condition=freq_condition,
            )
        else:
            H_pred = self.predictor(H_context=H_ctx, target_pos=target_pos)

        # 6. Target Representation & Unified Regularization
        use_target_ema = getattr(self.config, "use_target_ema", False)
        if use_target_ema:
            with torch.no_grad():
                H_tgt = self.target_encoder(target_tokens, target_pos, target_indices=target_indices)
            H_rep_reg = H_ctx
        else:
            # Single Shared Encoder + Stop-Gradient Target (SimSiam/VICReg hybrid)
            # H_tgt_full has active gradients for VICReg representation regularization
            # H_tgt is detached for prediction loss to prevent chasing collapse
            H_tgt_full = self.context_encoder(target_tokens, target_pos, context_indices=target_indices)
            H_tgt = H_tgt_full.detach()
            H_rep_reg = torch.cat([H_ctx, H_tgt_full], dim=1)

        # 7a. Compute Lift-Off Perturbation (if liftoff_invar_weight > 0)
        H_ctx_pert = None
        if self.training and getattr(self.config, "liftoff_invar_weight", 0.0) > 0.0:
            alpha = 0.70 + 0.25 * torch.rand(B, 1, 1, 1, device=device, dtype=x.dtype)
            x_fft_full = torch.fft.rfft(x.float(), dim=-1)
            num_bins = x_fft_full.shape[-1]
            decay = torch.linspace(1.0, 0.85, num_bins, device=device, dtype=x.dtype).view(1, 1, 1, -1)
            x_pert_fft = x_fft_full * alpha * decay
            x_pert = torch.fft.irfft(x_pert_fft, n=self.config.in_channels, dim=-1).to(x.dtype)
            tokens_pert, _ = self.tokenizer(x_pert)
            context_tokens_pert = tokens_pert[batch_arange, context_indices]
            H_ctx_pert = self.context_encoder(context_tokens_pert, context_pos)

        # 7b. Compute Energy-Weighted Spectral Phase for Phase-Depth Alignment (if phase_align_weight > 0)
        z_depth_tgt = None
        z_depth_pred = None
        phase_tgt = None
        z_depth = None
        phase_ctx = None
        if getattr(self.config, "phase_align_weight", 0.0) > 0.0:
            phi_ewp = DualScaleDiffusionTokenizer5x5.compute_energy_weighted_phase(
                x, num_bins=getattr(self.config, "num_freq_bins", 14)
            )  # [B, 25] in [-1, 1]
            if N_total == 100:
                # Isolate deepest late diffusion stage tau=3 for target tokens
                mask_t3 = (target_indices % 4 == 3)  # [B, N_tgt]
                k_t3 = mask_t3[0].sum().item()
                if k_t3 > 0:
                    H_tgt_t3 = H_tgt[mask_t3].view(B, k_t3, -1)
                    H_pred_t3 = H_pred[mask_t3].view(B, k_t3, -1)
                    z_depth_tgt = self.depth_head(H_tgt_t3).squeeze(-1)   # [B, 8]
                    z_depth_pred = self.depth_head(H_pred_t3).squeeze(-1) # [B, 8]
                    spatial_tgt_idx = target_indices[mask_t3].view(B, k_t3) // 4  # [B, 8] in 0..24
                    phase_tgt = torch.gather(phi_ewp, dim=1, index=spatial_tgt_idx) # [B, 8]
            else:
                spatial_idx = (context_indices // 2) if N_total == 50 else context_indices
                phase_ctx = torch.gather(phi_ewp, dim=1, index=spatial_idx)
                z_depth = self.depth_head(H_ctx).squeeze(-1)

        # 7c. Compute Combined JEPA Loss
        loss_dict = self.loss_fn(
            H_pred=H_pred,
            H_target=H_tgt,
            target_indices=target_indices,
            H_ctx=H_ctx,
            H_ctx_pert=H_ctx_pert,
            z_depth_tgt=z_depth_tgt,
            z_depth_pred=z_depth_pred,
            phase_tgt=phase_tgt,
            z_depth=z_depth,
            phase_ctx=phase_ctx,
            x_raw=x,
            delta_pred=delta_pred,
            H_rep_reg=H_rep_reg,
        )
        loss_dict.update({
            "H_pred": H_pred,
            "H_tgt": H_tgt,
            "H_ctx": H_ctx,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "mask_bool": mask_bool
        })
        return loss_dict

    @torch.no_grad()
    def extract_center_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference feature extraction for center point of the grid (all points visible).
        Input: [B, 5, 5, C] -> Output: [B, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        tokens, pos = self.tokenizer(x)
        H = self.context_encoder(tokens, pos)  # [B, N_total, D]

        if getattr(self.config, "spatial_topology", "dense_5x5") in ("concentric_star", "star", "octagram"):
            center_spatial_idx = 0
        else:
            center_spatial_idx = (self.config.grid_size // 2) * self.config.grid_size + (self.config.grid_size // 2)

        if tokens.shape[1] == 100:
            s_tok = center_spatial_idx * 4
            return H[:, s_tok : s_tok + 4, :].mean(dim=1)  # [B, D]
        elif tokens.shape[1] == 50:
            s_tok = center_spatial_idx * 2
            return 0.5 * (H[:, s_tok, :] + H[:, s_tok + 1, :])  # [B, D]
        else:
            return H[:, center_spatial_idx, :]  # [B, D]

    @torch.no_grad()
    def extract_unified_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dual-Perspective Representation Extraction:
        Concatenates Context Encoder representation H_ctx with the JEPA
        physical prediction discrepancy vector Delta_H = |H_target - H_pred|.
        
        Input: [B, 5, 5, C] -> Output: [B, 2 * D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        device = x.device

        # 1. Full tokenization
        tokens, pos = self.tokenizer(x)
        N_total = tokens.shape[1]

        # 2. Context features from visible tokens
        H_full = self.context_encoder(tokens, pos)  # [B, N_total, D]

        if getattr(self.config, "spatial_topology", "dense_5x5") in ("concentric_star", "star", "octagram"):
            center_spatial_idx = 0
        else:
            center_spatial_idx = (self.config.grid_size // 2) * self.config.grid_size + (self.config.grid_size // 2)

        if N_total == 100:
            s_tok = center_spatial_idx * 4
            h_ctx_center = H_full[:, s_tok : s_tok + 4, :].mean(dim=1)  # [B, D]
            center_tgt_indices = torch.arange(s_tok, s_tok + 4, device=device)
        elif N_total == 50:
            s_tok = center_spatial_idx * 2
            h_ctx_center = 0.5 * (H_full[:, s_tok, :] + H_full[:, s_tok + 1, :])  # [B, D]
            center_tgt_indices = torch.tensor([s_tok, s_tok + 1], device=device)
        else:
            h_ctx_center = H_full[:, center_spatial_idx, :]  # [B, D]
            center_tgt_indices = torch.tensor([center_spatial_idx], device=device)

        # 3. Context & Target Mask for JEPA Prediction Discrepancy
        batch_arange = torch.arange(B, device=device).unsqueeze(1)
        all_indices = torch.arange(N_total, device=device)
        mask_tgt = torch.zeros(N_total, dtype=torch.bool, device=device)
        mask_tgt[center_tgt_indices] = True

        ctx_idx = all_indices[~mask_tgt].unsqueeze(0).expand(B, -1)  # [B, N_ctx]
        tgt_idx = center_tgt_indices.unsqueeze(0).expand(B, -1)      # [B, N_tgt]

        ctx_tokens = tokens[batch_arange, ctx_idx]
        ctx_pos = pos[batch_arange, ctx_idx]
        target_pos = pos[batch_arange, tgt_idx]

        H_ctx_masked = self.context_encoder(ctx_tokens, ctx_pos, context_indices=ctx_idx)

        # Predict center target tokens from surrounding context
        if hasattr(self.predictor, "residual_head"):
            freq_cond = self.compute_characteristic_frequency(
                x, num_bins=getattr(self.config, "num_freq_bins", 14)
            )
            H_pred = self.predictor(
                H_context=H_ctx_masked,
                target_pos=target_pos,
                context_indices=ctx_idx,
                target_indices=tgt_idx,
                freq_condition=freq_cond,
            )
        else:
            H_pred = self.predictor(
                H_context=H_ctx_masked,
                target_pos=target_pos,
                context_indices=ctx_idx,
                target_indices=tgt_idx,
            )

        # Target representation
        target_tokens = tokens[batch_arange, tgt_idx]
        if getattr(self.config, "use_target_ema", False):
            H_tgt = self.target_encoder(target_tokens, target_pos, target_indices=tgt_idx)
        else:
            H_tgt = self.context_encoder(target_tokens, target_pos, context_indices=tgt_idx)

        # Coordinate-wise physical discrepancy: mean across center target tokens
        delta_H = torch.abs(H_tgt - H_pred).mean(dim=1)  # [B, D]

        # Concatenate into unified [B, 2 * D] vector
        Z_unified = torch.cat([h_ctx_center, delta_H], dim=-1)  # [B, 2 * D]
        return Z_unified

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unified feature extractor dispatching according to config.feature_extraction_mode.
        'unified': [B, 2 * D] dual-perspective representation
        'context': [B, D] context encoder center feature
        """
        mode = getattr(self.config, "feature_extraction_mode", "unified")
        if mode == "unified":
            return self.extract_unified_features(x)
        return self.extract_center_feature(x)

    @torch.no_grad()
    def extract_all_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract all 25 latent features for the 5x5 grid.
        Input: [B, 5, 5, C] -> Output: [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        tokens, pos = self.tokenizer(x)
        H = self.context_encoder(tokens, pos)  # [B, N_total, D]
        if tokens.shape[1] == 100:
            # Average 4 temporal stages for each of the 25 spatial points
            return H.view(-1, 25, 4, H.shape[-1]).mean(dim=2)  # [B, 25, D]
        elif tokens.shape[1] == 50:
            # Average shallow and deep tokens for each of the 25 spatial points
            return 0.5 * (H[:, 0::2, :] + H[:, 1::2, :])  # [B, 25, D]
        else:
            return H  # [B, 25, D]

    @torch.no_grad()
    def extract_attention_map(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract self-attention weights from the final Context Encoder block.
        Input: [B, 5, 5, C] -> Output: [B, num_heads, N_total, N_total]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        tokens, pos = self.tokenizer(x)
        _, attn = self.context_encoder(tokens, pos, return_attention=True)
        return attn
