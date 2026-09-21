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
from .tokenizer_5x5 import SpatialGridTokenizer5x5, DualDomainGridTokenizer5x5, DualDomainAttentionTokenizer5x5, build_tokenizer_5x5
from .context_encoder import ContextEncoder5x5
from .target_encoder import TargetEncoder5x5
from .predictor import Predictor5x5, OperatorDiffusionPredictor5x5, build_predictor_5x5
from ..masking.cluster_mask import ContiguousClusterMasker5x5
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

        # 1. Tokenizer (Dual-Domain Spatiotemporal-Spectral or Time-Only)
        self.tokenizer = build_tokenizer_5x5(config)

        # 2. Contiguous Cluster Masker
        self.masker = ContiguousClusterMasker5x5(
            min_masked=config.min_masked,
            max_masked=config.max_masked,
            grid_size=config.grid_size
        )

        # 3. Context Encoder
        self.context_encoder = ContextEncoder5x5(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )

        # 4. Target Encoder (EMA)
        self.target_encoder = TargetEncoder5x5(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )
        self._init_target_encoder()

        # 5. Predictor (Physics Operator Diffusion Predictor or Standard)
        self.predictor = build_predictor_5x5(config)

        # 6. Loss Function with Physics-Informed Regularization
        self.loss_fn = JEPALoss5x5(
            loss_type=config.loss_type,
            eps=config.eps,
            liftoff_invar_weight=getattr(config, "liftoff_invar_weight", 0.0),
            phase_align_weight=getattr(config, "phase_align_weight", 0.0),
            var_weight=config.var_weight,
            cov_weight=config.cov_weight,
            var_gamma=config.var_gamma,
            vicreg_target=getattr(config, "vicreg_target", "context"),
            rank_barrier_weight=getattr(config, "rank_barrier_weight", 0.0),
            rank_barrier_eps=getattr(config, "rank_barrier_eps", 1e-4),
        )

        # 7. Physical Alignment Modules (Option B)
        self.depth_head = nn.Linear(config.embed_dim, 1, bias=False)
        nn.init.trunc_normal_(self.depth_head.weight, std=0.02)

        self.target_harmonic_proj = nn.Sequential(
            nn.Linear(2, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

    def _init_target_encoder(self):
        """Copy initial weights from Context Encoder to Target Encoder."""
        for p_tgt, p_ctx in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad = False

    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update Target Encoder weights via EMA."""
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

        # 1. Tokenization -> tokens [B, 25, D], pos [B, 25, D]
        tokens, pos = self.tokenizer(x)

        # 2. Sample or use custom mask
        if custom_context_indices is not None and custom_target_indices is not None:
            context_indices = custom_context_indices.to(device)
            target_indices = custom_target_indices.to(device)
            mask_bool = torch.zeros(B, self.config.grid_size * self.config.grid_size, dtype=torch.bool, device=device)
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
        H_ctx = self.context_encoder(context_tokens, context_pos)

        # 5. Predictor (Physics Operator Diffusion Predictor or Standard)
        is_operator_diff = hasattr(self.predictor, "op_embedding") and getattr(self.config, "predictor_type", "") == "operator_diffusion"
        if is_operator_diff:
            if freq_condition is None:
                if self.training:
                    freq_condition = torch.randint(1, getattr(self.config, "num_freq_bins", 14) + 1, (B,), device=device)
                else:
                    freq_condition = None
            H_pred = self.predictor(H_context=H_ctx, target_pos=target_pos, freq_condition=freq_condition)
        else:
            H_pred = self.predictor(H_context=H_ctx, target_pos=target_pos)

        # 6. Target Encoder (EMA, detached)
        with torch.no_grad():
            H_tgt_base = self.target_encoder(target_tokens, target_pos)

        # 6b. Frequency-Resolved Target for Operator Diffusion JEPA
        if is_operator_diff and freq_condition is not None:
            # Extract harmonic feature (phase & log-magnitude) of harmonic k at target locations
            x_flat = x.reshape(B, self.config.grid_size * self.config.grid_size, self.config.in_channels)
            x_tgt_raw = x_flat[batch_arange, target_indices]  # [B, N_tgt, C]
            X_tgt_fft = torch.fft.rfft(x_tgt_raw.float(), dim=-1)  # [B, N_tgt, C//2 + 1]
            k_expand = freq_condition.view(B, 1, 1).expand(-1, target_indices.shape[1], 1)
            X_k = torch.gather(X_tgt_fft, dim=-1, index=k_expand).squeeze(-1)  # [B, N_tgt]
            phase_k = torch.angle(X_k) / torch.pi
            mag_k = torch.log1p(torch.abs(X_k))
            harm_feat = torch.stack([phase_k, mag_k], dim=-1)  # [B, N_tgt, 2]
            harm_emb = self.target_harmonic_proj(harm_feat)  # [B, N_tgt, D]
            H_tgt = F.layer_norm(H_tgt_base + harm_emb, (self.config.embed_dim,))
        else:
            H_tgt = H_tgt_base

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

        # 7b. Compute Fundamental Harmonic Phase for Phase-Depth Alignment (if phase_align_weight > 0)
        z_depth = None
        phase_ctx = None
        if getattr(self.config, "phase_align_weight", 0.0) > 0.0:
            x_flat = x.reshape(B, self.config.grid_size * self.config.grid_size, self.config.in_channels)
            X_fft = torch.fft.rfft(x_flat.float(), dim=-1)
            phase_1 = torch.angle(X_fft[:, :, 1]) / torch.pi  # [B, 25] in [-1, 1]
            phase_ctx = phase_1[batch_arange, context_indices]  # [B, N_ctx]
            z_depth = self.depth_head(H_ctx).squeeze(-1)  # [B, N_ctx]

        # 7c. Compute Combined JEPA Loss
        loss_dict = self.loss_fn(
            H_pred=H_pred,
            H_target=H_tgt,
            H_ctx=H_ctx,
            H_ctx_pert=H_ctx_pert,
            z_depth=z_depth,
            phase_ctx=phase_ctx,
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
        Inference feature extraction for center point (2, 2) of the 5x5 grid (all points visible).
        Input: [B, 5, 5, C] -> Output: [B, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        tokens, pos = self.tokenizer(x)
        H = self.context_encoder(tokens, pos)  # [B, 25, D]
        center_idx = (self.config.grid_size // 2) * self.config.grid_size + (self.config.grid_size // 2)  # index 12
        return H[:, center_idx, :]  # [B, D]

    @torch.no_grad()
    def extract_all_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract all 25 latent features for the 5x5 grid.
        Input: [B, 5, 5, C] -> Output: [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        tokens, pos = self.tokenizer(x)
        return self.context_encoder(tokens, pos)

    @torch.no_grad()
    def extract_attention_map(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract self-attention weights from the final Context Encoder block.
        Input: [B, 5, 5, C] -> Output: [B, num_heads, 25, 25]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        tokens, pos = self.tokenizer(x)
        _, attn = self.context_encoder(tokens, pos, return_attention=True)
        return attn
