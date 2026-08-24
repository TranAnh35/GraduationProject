"""
Complete 1D Temporal PECT-JEPA Model Architecture (implement.md, Section 4.8).
Integrates 1D Tokenizer, Dynamic 1D Masker, Context Encoder 1D, EMA Target Encoder 1D, Predictor 1D, and Normalized L2 Loss.
"""

import copy
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn

from ..configs.config import Temporal1DConfig, get_default_config_1d
from .tokenizer_1d import TemporalTokenizer1D
from .context_encoder_1d import ContextEncoder1D
from .target_encoder_1d import TargetEncoder1D
from .predictor_1d import Predictor1D
from ..masking.temporal_mask import Dynamic1DBlockMasker
from ..losses.jepa_loss import JEPALoss1D


class PECT_JEPA_1D(nn.Module):
    """
    Complete 1D Temporal PECT-JEPA Self-Supervised Model (TS-JEPA).
    Input: [B, T=500] 1D transient waveform.
    """
    def __init__(self, config: Optional[Temporal1DConfig] = None):
        super().__init__()
        if config is None:
            config = get_default_config_1d()
        self.config = config

        # Module 1: 1D Temporal Tokenizer
        self.tokenizer = TemporalTokenizer1D(
            time_samples=config.time_samples,
            patch_length=config.patch_length,
            stride=config.stride,
            embed_dim=config.embed_dim,
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout
        )

        # Module 2: Dynamic 1D Temporal Masker
        self.masker = Dynamic1DBlockMasker(
            num_patches=config.num_patches,
            mask_strategy=config.mask_strategy,
            mask_ratio=config.mask_ratio,
            num_visible_early=config.num_visible_early
        )

        # Module 3: 1D Context Encoder
        self.context_encoder = ContextEncoder1D(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )

        # Module 4: 1D EMA Target Encoder
        self.target_encoder = TargetEncoder1D(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )
        self._init_target_encoder()

        # Module 5: 1D Predictor
        self.predictor = Predictor1D(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )

        # Module 6: JEPA Loss
        self.loss_fn = JEPALoss1D(
            loss_type=config.loss_type,
            eps=config.eps
        )

    def _init_target_encoder(self):
        """Initialize Target Encoder parameters to match Context Encoder."""
        for param_t, param_c in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False

    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update Target Encoder parameters via Exponential Moving Average."""
        if momentum is None:
            momentum = self.config.ema_momentum
        self.target_encoder.update_ema(self.context_encoder, momentum=momentum)

    def forward(
        self,
        x: torch.Tensor,
        custom_context_indices: Optional[torch.Tensor] = None,
        custom_target_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through 1D Temporal PECT-JEPA.

        Args:
            x: Input 1D waveform tensor [B, T=500] (or [T=500])

        Returns:
            Dict containing:
                - 'loss': scalar prediction loss
                - 'H_pred': [B, N_tgt, D] predicted target latents
                - 'H_target': [B, N_tgt, D] target latents from EMA Target Encoder
                - 'H_context': [B, N_ctx, D] encoded context latents
                - 'context_indices': [B, N_ctx]
                - 'target_indices': [B, N_tgt]
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, T]

        B = x.shape[0]
        device = x.device

        # 1. 1D Tokenizer -> tokens [B, 16, D], pos_embed [1, 16, D]
        tokens, pos_embed = self.tokenizer(x)

        # 2. Dynamic 1D Masking -> context_indices, target_indices
        if custom_context_indices is not None and custom_target_indices is not None:
            context_indices = custom_context_indices
            target_indices = custom_target_indices
        else:
            context_indices, target_indices, _ = self.masker.sample_mask(
                batch_size=B,
                device=device
            )

        # 3. Context & Target Token Partitioning (Token Dropping)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        pos_expanded = pos_embed.expand(B, -1, -1)

        # Context tokens and 1D positions
        context_tokens = tokens[batch_idx, context_indices]  # [B, N_ctx, D]
        context_pos = pos_expanded[batch_idx, context_indices]  # [B, N_ctx, D]

        # Target tokens and 1D positions
        target_tokens = tokens[batch_idx, target_indices]  # [B, N_tgt, D]
        target_pos = pos_expanded[batch_idx, target_indices]  # [B, N_tgt, D]

        # 4. Context Encoder (Encodes ONLY visible context patches)
        H_context = self.context_encoder(
            context_tokens=context_tokens,
            context_pos=context_pos
        )  # [B, N_ctx, D]

        # 5. Predictor (Predicts target latents from H_context and target_pos)
        H_pred = self.predictor(
            H_context=H_context,
            target_pos=target_pos
        )  # [B, N_tgt, D]

        # 6. EMA Target Encoder (Encodes target patches, detached)
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
            "context_indices": context_indices,
            "target_indices": target_indices
        }

    @torch.no_grad()
    def extract_features(
        self,
        x: torch.Tensor,
        pool: bool = False
    ) -> torch.Tensor:
        """
        Extract latent representations for downstream evaluation.
        Operates on frozen encoder with all 16 patches visible.

        Args:
            x: Input 1D waveform [B, T=500] (or [T=500])
            pool: Whether to average over the 16 patches (default: False)

        Returns:
            If pool=False: [B, 16, D]
            If pool=True:  [B, D]
        """
        orig_ndim = x.ndim
        if orig_ndim == 1:
            x = x.unsqueeze(0)

        B = x.shape[0]
        # 1. Tokenization
        tokens, pos_embed = self.tokenizer(x)

        # 2. Context Encoder with all patches visible
        H_all = self.context_encoder(
            context_tokens=tokens,
            context_pos=pos_embed.expand(B, -1, -1)
        )  # [B, 16, D]

        if pool:
            H_all = torch.mean(H_all, dim=1)  # [B, D]

        if orig_ndim == 1:
            H_all = H_all.squeeze(0)

        return H_all
