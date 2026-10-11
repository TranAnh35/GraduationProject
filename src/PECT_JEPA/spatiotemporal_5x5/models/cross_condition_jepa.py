"""
Cross-Condition Paired JEPA Architecture for PECT.
Core Idea:
- Inputs: Pair (x_A, x_B) at identical physical plate coordinates from two distinct measurement conditions.
- Context: Visible outer ring tokens from Condition A.
- Target: Masked core tokens from Condition B.
- Predictor: Predicts target latent of Condition B given context of Condition A + Condition B metadata.
- Learns invariant material physics (underlying flaw structure) rather than sensor/liftoff quirks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..configs.config import Spatiotemporal5x5Config, get_default_config_5x5
from .tokenizer_5x5 import build_tokenizer_5x5
from .context_encoder import build_context_encoder_5x5
from .target_encoder import TargetEncoder5x5
from .predictor import build_predictor_5x5
def compute_vicreg_loss(z: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 1.0, gamma: float = 1.0, eps: float = 1e-4):
    """Standard VICReg coordinate-wise variance hinge and covariance decorrelation penalty."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = torch.mean(F.relu(gamma - std))

    z_cent = z - z.mean(dim=0, keepdim=True)
    N = z.shape[0]
    cov = (z_cent.T @ z_cent) / max(N - 1, 1)
    cov_off = cov - torch.diag(torch.diag(cov))
    cov_loss = (cov_off ** 2).sum() / z.shape[1]

    total = var_weight * var_loss + cov_weight * cov_loss
    return total, var_loss, cov_loss


class ConditionEmbedding(nn.Module):
    """
    Encodes sensor, waveform, and lift-off condition IDs into condition embedding vector e in R^D.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.sensor_embed = nn.Embedding(4, embed_dim)
        self.waveform_embed = nn.Embedding(4, embed_dim)
        self.liftoff_embed = nn.Embedding(4, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, cond_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond_b: [B, 3] containing [sensor_id, waveform_id, liftoff_id]
        Returns:
            e: [B, 1, embed_dim]
        """
        s = self.sensor_embed(cond_b[:, 0])
        w = self.waveform_embed(cond_b[:, 1])
        l = self.liftoff_embed(cond_b[:, 2])
        e = s + w + l
        return self.proj(e).unsqueeze(1)


class CrossConditionJEPA(nn.Module):
    """
    Cross-Condition Paired JEPA Model.
    """
    def __init__(self, config: Optional[Spatiotemporal5x5Config] = None):
        super().__init__()
        if config is None:
            config = get_default_config_5x5()
        self.config = config

        # 1. Tokenizer (shared between A and B)
        self.tokenizer = build_tokenizer_5x5(config)

        # 2. Context Encoder (encodes x_A)
        self.context_encoder = build_context_encoder_5x5(config)

        # 4. Target Encoder (encodes x_B, stop-gradient)
        self.target_encoder = TargetEncoder5x5(
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            encoder_type=getattr(config, "encoder_type", "dispersion_conditioned"),
            spatial_topology=getattr(config, "spatial_topology", "concentric_star"),
            star_radii=getattr(config, "star_radii", (1, 3, 7)),
            grid_size=config.grid_size,
        )
        self._init_target_encoder()

        # 5. Condition Embedding of target Condition B
        self.cond_embed = ConditionEmbedding(embed_dim=config.embed_dim)

        # 6. Predictor
        self.predictor = build_predictor_5x5(config)

        # Default mask: 9 core probes as target (indices 0..8)
        # 16 outer probes as context (indices 9..24)
        self.register_buffer("default_target_idx", torch.arange(9, dtype=torch.long))
        self.register_buffer("default_context_idx", torch.arange(9, 25, dtype=torch.long))

    def _init_target_encoder(self):
        for p_tgt, p_ctx in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad = False

    def update_target_encoder(self, momentum: float = 0.996):
        with torch.no_grad():
            for p_tgt, p_ctx in zip(self.target_encoder.encoder.parameters(), self.context_encoder.parameters()):
                p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        cond_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward step:
        x_a: [B, 5, 5, C] measurement condition A
        x_b: [B, 5, 5, C] measurement condition B at SAME coordinates
        cond_b: [B, 3] metadata of condition B
        """
        B = x_a.shape[0]
        device = x_a.device

        # 1. Tokenize both conditions
        tokens_a, pos_a = self.tokenizer(x_a) # [B, 25, D]
        tokens_b, pos_b = self.tokenizer(x_b) # [B, 25, D]

        # 2. Partition: Context from A (16 outer), Target from B (9 core)
        ctx_idx = self.default_context_idx.unsqueeze(0).expand(B, -1) # [B, 16]
        tgt_idx = self.default_target_idx.unsqueeze(0).expand(B, -1) # [B, 9]

        batch_arange = torch.arange(B, device=device).unsqueeze(1)

        ctx_tokens_a = tokens_a[batch_arange, ctx_idx] # [B, 16, D]
        ctx_pos_a = pos_a[batch_arange, ctx_idx]

        tgt_tokens_b = tokens_b[batch_arange, tgt_idx] # [B, 9, D]
        tgt_pos_b = pos_b[batch_arange, tgt_idx]

        # 3. Context Encoder encodes visible context from A
        H_ctx_a = self.context_encoder(
            ctx_tokens_a,
            ctx_pos_a,
            context_indices=ctx_idx
        ) # [B, 16, D]

        # 4. Target Encoder encodes target from B (detached)
        with torch.no_grad():
            H_tgt_b = self.target_encoder(
                tgt_tokens_b,
                tgt_pos_b,
                target_indices=tgt_idx
            ) # [B, 9, D]

        # 5. Condition Query: condition B embedding added to target queries
        e_b = self.cond_embed(cond_b) # [B, 1, D]
        conditioned_tgt_pos = tgt_pos_b + e_b

        # 6. Predictor predicts H_tgt_b from H_ctx_a + conditioned target queries
        H_pred_b = self.predictor(
            H_context=H_ctx_a,
            target_pos=conditioned_tgt_pos,
            context_indices=ctx_idx,
            target_indices=tgt_idx,
        ) # [B, 9, D]

        # 7. Prediction Loss (L1)
        pred_loss = F.l1_loss(H_pred_b, H_tgt_b)

        # 8. VICReg Regularization on H_ctx_a representations
        flat_rep = H_ctx_a.reshape(B * H_ctx_a.shape[1], -1)
        vicreg_loss, var_loss, cov_loss = compute_vicreg_loss(
            flat_rep,
            var_weight=getattr(self.config, "var_weight", 1.0),
            cov_weight=getattr(self.config, "cov_weight", 1.0),
            gamma=1.0,
            eps=1e-4,
        )

        total_loss = pred_loss + vicreg_loss

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "H_ctx_a": H_ctx_a,
            "H_tgt_b": H_tgt_b,
            "H_pred_b": H_pred_b,
        }

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts robust representation for a single scan patch during inference.
        Returns unified [B, 2*D] representation concatenating center token with contextual pool.
        """
        tokens, pos = self.tokenizer(x) # [B, 25, D]
        h_all = self.context_encoder(tokens, pos) # [B, 25, D]
        h_center = h_all[:, 0, :] # [B, D] Center probe
        h_context = torch.mean(h_all[:, 1:, :], dim=1) # [B, D] Outer context pool
        return torch.cat([h_center, h_context], dim=-1) # [B, 2*D = 128D]
