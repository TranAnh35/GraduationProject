"""
PECT-JEPA Model Architecture for v0.2 (Section 2, implement.md).
Integrates Frame-by-Frame Tokenizer, Dynamic Frame-by-Frame Block Masking,
Token-Dropping Context Encoder, EMA Target Encoder, Predictor, and JEPA Loss.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any

from ..configs.config import PECTJEPAConfig
from .tokenizer import SpatioTemporalTokenizer
from .context_encoder import ContextEncoder
from .target_encoder import TargetEncoder
from .predictor import Predictor
from ..masking.spatiotemporal_mask import DynamicSpatioTemporalBlockMasker
from ..losses.jepa_loss import JEPALoss


class PECT_JEPA(nn.Module):
    """
    Complete PECT-JEPA Self-Supervised Model (v0.2).
    Input: [B, H, W, T_c] (e.g. [B, 300, 300, 16]).
    """
    def __init__(self, config: Optional[PECTJEPAConfig] = None):
        super().__init__()
        if config is None:
            from ..configs.config import get_default_config
            config = get_default_config()
        self.config = config

        # Module 1: Spatio-Temporal Tokenizer (frame-by-frame)
        self.tokenizer = SpatioTemporalTokenizer(
            spatial_patch=config.tokenizer.spatial_patch,
            clip_length=config.clip.temporal_length,
            embed_dim=config.tokenizer.embed_dim,
            pos_embed_type=config.tokenizer.pos_embed_type,
            dropout=config.tokenizer.dropout
        )

        # Module 2: Dynamic Frame-by-Frame Block Masker
        self.masker = DynamicSpatioTemporalBlockMasker(
            spatial_block_h=config.mask.spatial_block_h,
            spatial_block_w=config.mask.spatial_block_w,
            num_masked_frames=getattr(config.mask, "num_masked_frames", 8),
            min_masked_frames=getattr(config.mask, "min_masked_frames", 4),
            max_masked_frames=getattr(config.mask, "max_masked_frames", 10)
        )

        # Module 3: Context Encoder (Token Dropping)
        self.context_encoder = ContextEncoder(
            embed_dim=config.tokenizer.embed_dim,
            depth=config.encoder.depth,
            num_heads=config.encoder.num_heads,
            mlp_ratio=config.encoder.mlp_ratio,
            dropout=config.encoder.dropout
        )

        # Module 4: EMA Target Encoder
        self.target_encoder = TargetEncoder(
            embed_dim=config.tokenizer.embed_dim,
            depth=config.encoder.depth,
            num_heads=config.encoder.num_heads,
            mlp_ratio=config.encoder.mlp_ratio,
            dropout=config.encoder.dropout
        )
        self._init_target_encoder()

        # Module 5: Predictor
        self.predictor = Predictor(
            embed_dim=config.tokenizer.embed_dim,
            depth=config.predictor.depth,
            num_heads=config.predictor.num_heads,
            mlp_ratio=config.predictor.mlp_ratio,
            dropout=config.predictor.dropout
        )

        # Module 6: JEPA Loss
        self.loss_fn = JEPALoss(
            loss_type=config.loss.loss_type,
            eps=config.loss.eps
        )

    def _init_target_encoder(self):
        """Initialize Target Encoder weights to match Context Encoder exactly."""
        for param_t, param_c in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False

    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update Target Encoder parameters using Exponential Moving Average."""
        if momentum is None:
            momentum = self.config.target_encoder.ema_momentum
        self.target_encoder.update_ema(self.context_encoder, momentum=momentum)

    def forward(
        self,
        clip: torch.Tensor,
        custom_context_indices: Optional[torch.Tensor] = None,
        custom_target_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PECT-JEPA (Section 2, implement.md).

        Args:
            clip: Input clip tensor [B, H, W, T_c] (e.g. [B, 300, 300, 16])

        Returns:
            Dict containing:
                - 'loss': scalar prediction loss
                - 'H_pred': [B, N_tgt, D] predicted target representations
                - 'H_target': [B, N_tgt, D] target representations from EMA Target Encoder
                - 'H_context': [B, N_ctx, D] encoded context representations
                - 'grid_shape': (H_t, W_t, T_c)
                - 'context_indices': [B, N_ctx]
                - 'target_indices': [B, N_tgt]
        """
        orig_ndim = clip.ndim
        if orig_ndim == 3:
            clip = clip.unsqueeze(0)

        B = clip.shape[0]
        device = clip.device

        # 1. Spatio-Temporal Tokenizer -> tokens [B, M, D], pos_embed [1, M, D], grid (H_t, W_t, T_c)
        tokens, pos_embed, grid_shape = self.tokenizer(clip)

        # 2. Dynamic Frame-by-Frame Block Masking -> context_indices, target_indices
        if custom_context_indices is not None and custom_target_indices is not None:
            context_indices = custom_context_indices
            target_indices = custom_target_indices
        else:
            context_indices, target_indices, _ = self.masker.sample_mask(
                batch_size=B,
                grid_shape=grid_shape,
                device=device
            )

        # 3. Context & Target Token Partitioning (Token Dropping)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        pos_expanded = pos_embed.expand(B, -1, -1)

        # Context tokens and 3D positions
        context_tokens = tokens[batch_idx, context_indices]  # [B, N_ctx, D]
        context_pos = pos_expanded[batch_idx, context_indices]  # [B, N_ctx, D]

        # Target tokens and 3D positions
        target_tokens = tokens[batch_idx, target_indices]  # [B, N_tgt, D]
        target_pos = pos_expanded[batch_idx, target_indices]  # [B, N_tgt, D]

        # 4. Context Encoder (Encodes ONLY visible context tokens)
        H_context = self.context_encoder(
            context_tokens=context_tokens,
            context_pos=context_pos
        )  # [B, N_ctx, D]

        # 5. Predictor (Predicts target latents from H_context and target_pos)
        H_pred = self.predictor(
            H_context=H_context,
            target_pos=target_pos
        )  # [B, N_tgt, D]

        # 6. EMA Target Encoder (Encodes target tokens, detached)
        H_target = self.target_encoder(
            target_tokens=target_tokens,
            target_pos=target_pos
        )  # [B, N_tgt, D]

        # 7. JEPA Loss
        loss = self.loss_fn(H_pred, H_target)

        return {
            "loss": loss,
            "H_pred": H_pred,
            "H_target": H_target,
            "H_context": H_context,
            "grid_shape": grid_shape,
            "context_indices": context_indices,
            "target_indices": target_indices
        }

    @torch.no_grad()
    def extract_features(
        self,
        clip: torch.Tensor,
        pool_temporal: bool = False
    ) -> torch.Tensor:
        """
        Extract full latent representations for downstream evaluation.
        Operates on frozen pretrained encoder with all tokens visible.

        Args:
            clip: Input clip tensor [B, H, W, T_c] (or [H, W, T_c])
            pool_temporal: Whether to average over temporal frames T_c (default: False)

        Returns:
            Latent token tensor:
                If pool_temporal=False: [B, H_t, W_t, T_c, D]
                If pool_temporal=True:  [B, H_t, W_t, D]
        """
        orig_ndim = clip.ndim
        if orig_ndim == 3:
            clip = clip.unsqueeze(0)

        B = clip.shape[0]
        # 1. Tokenization
        tokens, pos_embed, (H_t, W_t, T_c) = self.tokenizer(clip)

        # 2. Context Encoder with all tokens visible
        H_all = self.context_encoder(
            context_tokens=tokens,
            context_pos=pos_embed.expand(B, -1, -1)
        )  # [B, M, D]

        # Reshape to 3D token grid: [B, H_t, W_t, T_c, D]
        grid_latents = H_all.view(B, H_t, W_t, T_c, self.config.tokenizer.embed_dim)

        if pool_temporal:
            grid_latents = torch.mean(grid_latents, dim=3)  # [B, H_t, W_t, D]

        if orig_ndim == 3:
            grid_latents = grid_latents.squeeze(0)

        return grid_latents
