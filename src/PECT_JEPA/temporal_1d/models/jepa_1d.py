"""
Complete 1D Temporal PECT-JEPA Model (Stage A, physics-customized).
Integrates the log-time two-channel Tokenizer, Multi-Strategy Masker,
Context Encoder, EMA Target Encoder, Predictor and anti-collapse JEPA Loss.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from ..configs.config import Temporal1DConfig, get_default_config_1d
from .tokenizer_1d import TemporalTokenizer1D
from .context_encoder_1d import ContextEncoder1D
from .target_encoder_1d import TargetEncoder1D
from .predictor_1d import Predictor1D
from ..masking.temporal_mask import MultiStrategy1DMasker
from ..losses.jepa_loss import JEPALoss1D


class PECT_JEPA_1D(nn.Module):
    """
    Physics-customized 1D Temporal PECT-JEPA Self-Supervised Model (TS-JEPA).

    Input: [B, 2, T'=128] two-channel log-time waveform
           (built by data.preprocessing.build_two_channel_input / PECT1DDataset).
    """

    def __init__(self, config: Optional[Temporal1DConfig] = None):
        super().__init__()
        if config is None:
            config = get_default_config_1d()
        self.config = config

        # Module 1: Two-channel Tokenizer (A2)
        # 'resampled' (B1/B2): log-time input [B, 2, log_time_samples=128]
        # 'raw' (B0 baseline): padded raw input [B, 2, raw_padded_length=512]
        tok_mode = getattr(config, "tokenizer_mode", "resampled")
        tok_len = (
            config.raw_padded_length if tok_mode == "raw" else config.log_time_samples
        )
        self.tokenizer = TemporalTokenizer1D(
            log_time_samples=tok_len,
            num_patches=config.num_patches,
            num_channels=config.num_channels,
            embed_dim=config.embed_dim,
            pos_embed_type=config.pos_embed_type,
            t_total_ms=config.t_total_ms,
            t_start_frac=config.t_start_frac,
            dropout=config.dropout,
            mode=tok_mode,
        )

        # Module 2: Multi-strategy physics-informed Masker (A3)
        self.masker = MultiStrategy1DMasker(
            num_patches=config.num_patches,
            strategy_probs=config.strategy_probs,
            num_visible=config.num_visible,
        )

        # Module 3: Context Encoder (transformer blocks reused)
        self.context_encoder = ContextEncoder1D(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )

        # Module 4: EMA Target Encoder
        self.target_encoder = TargetEncoder1D(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )
        self._init_target_encoder()

        # Module 5: Predictor (mask-token queries + cross-attention)
        self.predictor = Predictor1D(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )

        # Module 6: JEPA Loss with anti-collapse (A4)
        self.loss_fn = JEPALoss1D(
            loss_type=config.loss_type,
            eps=config.eps,
            var_weight=config.var_weight,
            cov_weight=config.cov_weight,
            var_gamma=config.var_gamma,
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
        custom_target_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 2, T'] two-channel log-time input (or [2, T'] single sample)
            custom_context_indices / custom_target_indices: optional [B, N] overrides

        Returns dict: loss (+ components), H_pred, H_target, H_context,
                      context_indices, target_indices, mask_1d, strategy_ids.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1, 2, T']
        B = x.shape[0]
        device = x.device

        # 1. Tokenization -> tokens [B, N, D], pos [B, N, D], [1, N, D]
        tokens, pos_expanded, pos_embed = self.tokenizer(x)

        # 2. Multi-strategy masking
        if custom_context_indices is not None and custom_target_indices is not None:
            context_indices = custom_context_indices.to(device)
            target_indices = custom_target_indices.to(device)
            mask_1d = torch.zeros(B, self.config.num_patches, dtype=torch.bool, device=device)
            mask_1d.scatter_(1, target_indices, True)
            strategy_ids = torch.full((B,), -1, dtype=torch.long, device=device)
        else:
            context_indices, target_indices, mask_1d, strategy_ids = self.masker.sample_mask(
                batch_size=B, device=device
            )

        # 3. Token partitioning (context tokens are fully dropped)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        context_tokens = tokens[batch_idx, context_indices]      # [B, N_ctx, D]
        context_pos = pos_expanded[batch_idx, context_indices]
        target_tokens = tokens[batch_idx, target_indices]        # [B, N_tgt, D]
        target_pos = pos_expanded[batch_idx, target_indices]

        # 4. Context Encoder (visible context only)
        H_context = self.context_encoder(
            context_tokens=context_tokens, context_pos=context_pos
        )

        # 5. Predictor (mask-token queries cross-attend to context)
        H_pred = self.predictor(H_context=H_context, target_pos=target_pos)

        # 6. EMA Target Encoder (detached)
        H_target = self.target_encoder(
            target_tokens=target_tokens, target_pos=target_pos
        )

        # 7. JEPA Loss + anti-collapse components
        loss_dict = self.loss_fn(H_pred, H_target)
        loss_dict.update({
            "H_pred": H_pred,
            "H_target": H_target,
            "H_context": H_context,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "mask_1d": mask_1d,
            "strategy_ids": strategy_ids,
        })
        return loss_dict

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        """
        Frozen-encoder features for downstream evaluation (all patches visible).

        Args:
            x: [B, 2, T'] (or [2, T'])
            pool: mean over patches -> [B, D]; else [B, N, D]
        """
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(0)
        B = x.shape[0]
        tokens, pos_expanded, _ = self.tokenizer(x)
        H = self.context_encoder(context_tokens=tokens, context_pos=pos_expanded)
        if pool:
            H = torch.mean(H, dim=1)
        if orig_ndim == 2:
            H = H.squeeze(0)
        return H
