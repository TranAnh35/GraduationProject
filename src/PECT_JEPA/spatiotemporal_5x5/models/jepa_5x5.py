"""
Unified 5x5 Spatiotemporal PECT-JEPA Model.
Integrates the SpatialGridTokenizer5x5, ContiguousClusterMasker5x5,
ContextEncoder5x5, TargetEncoder5x5, Predictor5x5, and JEPALoss5x5.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from ..configs.config import Spatiotemporal5x5Config, get_default_config_5x5
from .tokenizer_5x5 import SpatialGridTokenizer5x5
from .context_encoder import ContextEncoder5x5
from .target_encoder import TargetEncoder5x5
from .predictor import Predictor5x5
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

        # 1. Tokenizer
        self.tokenizer = SpatialGridTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout
        )

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

        # 5. Predictor
        self.predictor = Predictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )

        # 6. Loss Function
        self.loss_fn = JEPALoss5x5(
            loss_type=config.loss_type,
            eps=config.eps,
            var_weight=config.var_weight,
            cov_weight=config.cov_weight,
            var_gamma=config.var_gamma,
            vicreg_target=getattr(config, "vicreg_target", "context"),
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward self-supervised step.
        Args:
            x: [B, 5, 5, C] input grid
            custom_context_indices: optional override [B, N_ctx]
            custom_target_indices:  optional override [B, N_tgt]

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

        # 5. Predictor (queries predict target representations)
        H_pred = self.predictor(H_context=H_ctx, target_pos=target_pos)

        # 6. Target Encoder (EMA, detached)
        with torch.no_grad():
            H_tgt = self.target_encoder(target_tokens, target_pos)

        # 7. JEPA Loss + Anti-collapse (C-JEPA regularizes H_ctx by default)
        loss_dict = self.loss_fn(H_pred, H_tgt, H_ctx=H_ctx)
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
