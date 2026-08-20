"""
PECT-JEPA Model Architecture (Section 5, 14, 24).
Integrates Multi-scale Temporal Encoder, Tokenizer, Dynamic Spatio-Temporal Masking,
Context Encoder, EMA Target Encoder, and Predictor.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any

from ..configs.config import PECTJEPAConfig
from .temporal_encoder import MultiScaleTemporalEncoder
from .tokenizer import SpatioTemporalTokenizer
from .context_encoder import ContextEncoder
from .target_encoder import TargetEncoder
from .predictor import Predictor
from ..masking.spatiotemporal_mask import DynamicSpatioTemporalBlockMasker
from ..losses.jepa_loss import JEPALoss


class PECT_JEPA(nn.Module):
    """
    Complete PECT-JEPA Self-Supervised Model.
    """
    def __init__(self, config: Optional[PECTJEPAConfig] = None):
        super().__init__()
        if config is None:
            from ..configs.config import get_default_config
            config = get_default_config()
        self.config = config

        # Module A: Multi-scale Temporal Encoder
        self.temporal_encoder = MultiScaleTemporalEncoder(
            raw_samples=config.temporal_encoder.raw_samples,
            t_prime=config.temporal_encoder.t_prime,
            kernel_sizes=config.temporal_encoder.conv_kernel_sizes,
            dilations=config.temporal_encoder.conv_dilations,
            hidden_dim=config.temporal_encoder.hidden_dim,
            transformer_blocks=config.temporal_encoder.transformer_blocks,
            transformer_heads=config.temporal_encoder.transformer_heads,
            dropout=config.temporal_encoder.dropout,
            out_dim=1
        )

        # Modules B & C: Temporal Clip & Spatio-Temporal Tokenizer
        self.tokenizer = SpatioTemporalTokenizer(
            spatial_patch=config.tokenizer.spatial_patch,
            clip_length=config.clip.temporal_length,
            stride=config.clip.stride,
            embed_dim=config.tokenizer.embed_dim,
            pos_embed_type=config.tokenizer.pos_embed_type,
            dropout=config.tokenizer.dropout
        )

        # Module D: Dynamic Spatio-Temporal Block Masker
        self.masker = DynamicSpatioTemporalBlockMasker(
            spatial_block_h=config.mask.spatial_block_h,
            spatial_block_w=config.mask.spatial_block_w,
            temporal_block_t=config.mask.temporal_block_t,
            num_blocks=config.mask.num_blocks,
            min_mask_ratio=config.mask.min_mask_ratio,
            max_mask_ratio=config.mask.max_mask_ratio
        )

        # Module F: Context Encoder
        self.context_encoder = ContextEncoder(
            embed_dim=config.tokenizer.embed_dim,
            depth=config.encoder.depth,
            num_heads=config.encoder.num_heads,
            mlp_ratio=config.encoder.mlp_ratio,
            dropout=config.encoder.dropout,
            attention_type=config.encoder.attention_type
        )

        # Module G: EMA Target Encoder
        self.target_encoder = TargetEncoder(
            embed_dim=config.tokenizer.embed_dim,
            depth=config.encoder.depth,
            num_heads=config.encoder.num_heads,
            mlp_ratio=config.encoder.mlp_ratio,
            dropout=config.encoder.dropout,
            attention_type=config.encoder.attention_type
        )
        # Initialize target encoder with context encoder weights
        self._init_target_encoder()

        # Module H: Predictor
        self.predictor = Predictor(
            embed_dim=config.tokenizer.embed_dim,
            depth=config.predictor.depth,
            num_heads=config.predictor.num_heads,
            mlp_ratio=config.predictor.mlp_ratio,
            dropout=config.predictor.dropout
        )

        # Loss function
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
        x: torch.Tensor,
        custom_context_indices: Optional[torch.Tensor] = None,
        custom_target_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PECT-JEPA (Section 14).

        Args:
            x: Input acquisition tensor [B, H, W, 500]

        Returns:
            Dict containing:
                - 'loss': scalar prediction loss
                - 'H_pred': [B, N_tgt, D] predicted target latents
                - 'H_target': [B, N_tgt, D] target latents from EMA Target Encoder
                - 'H_context': [B, N_ctx, D] encoded context latents
                - 'grid_shape': (H_t, W_t, K_t)
        """
        B = x.shape[0]
        device = x.device

        # 1. Multi-scale Temporal Encoder -> [B, H, W, T']
        x_latent = self.temporal_encoder(x)

        # 2. Spatio-Temporal Tokenizer -> [B, M, D], [1, M, D], (H_t, W_t, K_t)
        tokens, pos_embed, grid_shape = self.tokenizer(x_latent)

        # 3. Dynamic Masking -> context_indices, target_indices
        if custom_context_indices is not None and custom_target_indices is not None:
            context_indices = custom_context_indices
            target_indices = custom_target_indices
        else:
            context_indices, target_indices, _ = self.masker.sample_mask(
                batch_size=B,
                grid_shape=grid_shape,
                device=device
            )

        # 4. Context & Target Token Partitioning
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        pos_expanded = pos_embed.expand(B, -1, -1)

        # Context tokens and positions
        context_tokens = tokens[batch_idx, context_indices]  # [B, N_ctx, D]
        context_pos = pos_expanded[batch_idx, context_indices]  # [B, N_ctx, D]

        # Target tokens and positions
        target_tokens = tokens[batch_idx, target_indices]  # [B, N_tgt, D]
        target_pos = pos_expanded[batch_idx, target_indices]  # [B, N_tgt, D]

        # 5. Context Encoder -> H_context (Supports factorized spatio-temporal attention)
        H_context = self.context_encoder(
            context_tokens=context_tokens,
            context_pos=context_pos,
            context_indices=context_indices,
            grid_shape=grid_shape
        )

        # 6. Predictor -> H_pred (Target token values are NOT passed to Predictor!)
        H_pred = self.predictor(H_context, target_pos)

        # 7. EMA Target Encoder -> H_target (Detached from gradients)
        H_target = self.target_encoder(
            target_tokens=target_tokens,
            target_pos=target_pos,
            target_indices=target_indices,
            grid_shape=grid_shape
        )

        # 8. Compute JEPA Loss
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
        x: torch.Tensor,
        pool_temporal: bool = False
    ) -> torch.Tensor:
        """
        Extract full latent representations for downstream evaluation (Section 17).
        Operates on frozen pretrained encoder with all tokens visible.

        Args:
            x: Input acquisition tensor [B, H, W, 500] (or [H, W, 500])
            pool_temporal: Whether to average over temporal clips K_t (default: False)

        Returns:
            Latent token tensor:
                If pool_temporal=False: [B, H_t, W_t, K_t, D]
                If pool_temporal=True:  [B, H_t, W_t, D]
        """
        orig_ndim = x.ndim
        if orig_ndim == 3:
            x = x.unsqueeze(0)

        B = x.shape[0]
        device = x.device

        # 1. Temporal Encoding
        x_latent = self.temporal_encoder(x)

        # 2. Tokenization
        tokens, pos_embed, (H_t, W_t, K_t) = self.tokenizer(x_latent)

        # 3. Context Encoder with all tokens visible (indices 0..M-1)
        total_tokens = H_t * W_t * K_t
        all_indices = torch.arange(total_tokens, device=device).unsqueeze(0).expand(B, -1)

        H_all = self.context_encoder(
            context_tokens=tokens,
            context_pos=pos_embed.expand(B, -1, -1),
            context_indices=all_indices,
            grid_shape=(H_t, W_t, K_t)
        )  # [B, M, D]

        # Reshape to 3D token grid: [B, H_t, W_t, K_t, D]
        grid_latents = H_all.view(B, H_t, W_t, K_t, self.config.tokenizer.embed_dim)

        if pool_temporal:
            grid_latents = torch.mean(grid_latents, dim=3)  # [B, H_t, W_t, D]

        if orig_ndim == 3:
            grid_latents = grid_latents.squeeze(0)

        return grid_latents
