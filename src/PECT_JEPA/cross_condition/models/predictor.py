"""
Predictor for Cross-Condition PECT-JEPA.
Predicts masked target representations of Condition B from Condition A context
and Condition B self-supervised transfer function condition token c_B.
"""

from typing import Optional
import torch
import torch.nn as nn
from .attention import MultiheadSelfAttention, MultiheadCrossAttention, MLP
from ..configs.config import CrossConditionConfig


class PredictorBlock(nn.Module):
    """
    Predictor Transformer Block:
    1. Self-Attention among Target Queries
    2. Cross-Attention from Target Queries to Context H_context
    3. Feed-Forward MLP
    Supports AdaLN (Adaptive LayerNorm) modulation from condition token c_B.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        modulation: str = "adaln",
    ):
        super().__init__()
        self.modulation = modulation
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiheadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

        if self.modulation == "adaln":
            # AdaLN maps condition vector c_B to scale and shift for norm_self and norm_mlp:
            # (gamma_self, beta_self, gamma_mlp, beta_mlp) in R^{4*D}
            self.ada_mod = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, 4 * embed_dim),
            )
            # Zero-initialize the modulation projection so it starts as exact identity
            nn.init.zeros_(self.ada_mod[1].weight)
            nn.init.zeros_(self.ada_mod[1].bias)

    def forward(
        self,
        target_queries: torch.Tensor,
        H_context: torch.Tensor,
        condition_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.modulation == "adaln" and condition_vec is not None:
            # condition_vec: [B, 1, D] -> [B, D]
            if condition_vec.ndim == 3:
                c = condition_vec.squeeze(1)
            else:
                c = condition_vec
            params = self.ada_mod(c)  # [B, 4*D]
            gamma_s, beta_s, gamma_m, beta_m = params.chunk(4, dim=-1)
            gamma_s = gamma_s.unsqueeze(1)  # [B, 1, D]
            beta_s = beta_s.unsqueeze(1)
            gamma_m = gamma_m.unsqueeze(1)
            beta_m = beta_m.unsqueeze(1)

            # 1. Modulated Self-Attention
            norm_q = self.norm_self(target_queries) * (1.0 + gamma_s) + beta_s
            q = target_queries + self.self_attn(norm_q)

            # 2. Cross-Attention to Context
            q = q + self.cross_attn(
                query=self.norm_cross_q(q),
                key_value=self.norm_cross_kv(H_context),
            )

            # 3. Modulated MLP
            norm_m = self.norm_mlp(q) * (1.0 + gamma_m) + beta_m
            q = q + self.mlp(norm_m)
            return q
        else:
            # Standard unmodulated forward
            q = target_queries + self.self_attn(self.norm_self(target_queries))
            q = q + self.cross_attn(
                query=self.norm_cross_q(q),
                key_value=self.norm_cross_kv(H_context),
            )
            q = q + self.mlp(self.norm_mlp(q))
            return q


class CrossConditionPredictor(nn.Module):
    """
    Cross-Condition Predictor Network:
    Input:
        H_context: [B, N_ctx, D] representation from Condition A.
        target_pos: [B, N_tgt, D] target position embeddings.
        condition_token: [B, 1, D] Condition B transfer function token.
    Output:
        H_pred: [B, N_tgt, D] predicted target representations.
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config
        self.modulation = getattr(config, "condition_modulation", "query_projection")

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # When Option A continuous_pos_embed is active, decouple predictor mask_token
        # query projection from tokenizer.pos_mlp so encoder coordinate embeddings
        # are trained purely by ContextEncoder without mask_token gradient pollution.
        if getattr(config, "continuous_pos_embed", False):
            self.target_pos_proj = nn.Linear(config.embed_dim, config.embed_dim)
            nn.init.eye_(self.target_pos_proj.weight)
            nn.init.zeros_(self.target_pos_proj.bias)
        else:
            self.target_pos_proj = None

        # Condition projection layer for query conditioning (only when condition token is active)
        if getattr(config, "use_condition_token", False) and self.modulation == "query_projection":
            self.cond_query_proj = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.GELU(),
                nn.Linear(config.embed_dim, config.embed_dim),
            )
        else:
            self.cond_query_proj = None

        self.blocks = nn.ModuleList([
            PredictorBlock(
                embed_dim=config.embed_dim,
                num_heads=config.predictor_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                modulation=self.modulation,
            )
            for _ in range(config.predictor_depth)
        ])
        self.subspace_norm = bool(getattr(config, "subspace_norm", False)) and bool(
            getattr(config, "latent_disentanglement", False)
        )
        if self.subspace_norm:
            self.carrier_dim = int(getattr(config, "carrier_dim", config.embed_dim // 2))
            self.scattering_dim = int(getattr(config, "scattering_dim", config.embed_dim - self.carrier_dim))
            self.norm = None
            self.norm_c = nn.LayerNorm(self.carrier_dim)
            self.norm_s = nn.LayerNorm(self.scattering_dim, bias=False)
        else:
            self.norm = nn.LayerNorm(config.embed_dim)
            self.norm_c = None
            self.norm_s = None

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        condition_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_tgt, D = target_pos.shape
        tgt_pos_q = self.target_pos_proj(target_pos.detach()) if self.target_pos_proj is not None else target_pos
        queries = self.mask_token.expand(B, N_tgt, -1) + tgt_pos_q

        H_context_kv = H_context
        cond_for_block = None
        if condition_token is not None:
            if self.modulation == "query_projection":
                # Condition B is projected into the query representations
                # Queries specify: "At position (r,c), under Condition B, what is the field?"
                # Key/Values remain strictly H_context (16 spatial probes), forcing attention to spatial diffusion!
                c_proj = self.cond_query_proj(condition_token) if self.cond_query_proj is not None else condition_token
                queries = queries + c_proj
            elif self.modulation == "cross_attention":
                if condition_token.ndim == 2:
                    c = condition_token.unsqueeze(1)
                else:
                    c = condition_token
                H_context_kv = torch.cat([H_context, c], dim=1)
            elif self.modulation == "additive":
                queries = queries + condition_token
            elif self.modulation == "adaln":
                cond_for_block = condition_token

        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context_kv, condition_vec=cond_for_block)

        if self.subspace_norm:
            q_c = self.norm_c(q[..., :self.carrier_dim])
            q_s = self.norm_s(q[..., self.carrier_dim:])
            return torch.cat([q_c, q_s], dim=-1)
        return self.norm(q)
