"""
PECT-JEPA v2-A: Factorized Representation Architecture with Decoupled Heads.
Extends PECT-JEPA without altering the underlying tokenizer (ContinuousLinearFieldTokenizer5x5).

Deconstructs latent representation h(p) in R^D into:
- z_inv in R^{d_inv}: Shared Physical-State Representation (flaw severity, geometry, conductivity)
- z_meas in R^{d_meas}: Measurement-Specific Condition Representation (excitation dynamics, sensor transfer)

Maintains full backward compatibility with PECT_JEPA_5x5 and EXP-28 weights.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config


class FactorizedProjectionHead(nn.Module):
    """
    Decoupled Projection Heads mapping backbone representation h in R^D into
    z_inv in R^{d_inv} and z_meas in R^{d_meas}.
    """
    def __init__(self, embed_dim: int = 64, d_inv: int = 48, d_meas: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_inv = d_inv
        self.d_meas = d_meas

        # 1. Invariant Head: Shared physical-state projection
        self.head_inv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, d_inv),
        )

        # 2. Measurement Head: Excitation/sensor private condition projection
        self.head_meas = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, d_meas),
        )

        # LayerNorms to anchor scale before downstream heads
        self.ln_inv = nn.LayerNorm(d_inv)
        self.ln_meas = nn.LayerNorm(d_meas)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: h [..., embed_dim]
        Output: z_inv [..., d_inv], z_meas [..., d_meas]
        """
        z_inv = self.ln_inv(self.head_inv(h))
        z_meas = self.ln_meas(self.head_meas(h))
        return z_inv, z_meas


class PECT_JEPA_v2(PECT_JEPA_5x5):
    """
    PECT-JEPA v2-A Foundation Model with Factorized Representation Architecture.
    """
    def __init__(self, config: Spatiotemporal5x5Config, d_inv: Optional[int] = None, d_meas: Optional[int] = None):
        super().__init__(config)
        self.d_inv = d_inv or getattr(config, "d_inv", int(config.embed_dim * 0.75))
        self.d_meas = d_meas or getattr(config, "d_meas", config.embed_dim - self.d_inv)

        # Decoupled projection head on visible tokens
        self.factorized_head = FactorizedProjectionHead(
            embed_dim=config.embed_dim,
            d_inv=self.d_inv,
            d_meas=self.d_meas,
        )

    def factorize_tokens(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Factorizes token sequence H into invariant and measurement representations.
        Input: H [B, N, D]
        Returns: Z_inv [B, N, d_inv], Z_meas [B, N, d_meas]
        """
        return self.factorized_head(H)

    @torch.no_grad()
    def extract_invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downstream zero-shot evaluation extractor.
        Extracts purely the shared physical-state representation z_inv for the center probe.
        Input: [B, 5, 5, C] -> Output: [B, d_inv]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        tokens, pos = self.tokenizer(x)
        H = self.context_encoder(tokens, pos)  # [B, 25, D]
        # Probe index 0 is center probe (0, 0)
        h_center = H[:, 0, :]
        z_inv, _ = self.factorized_head(h_center)
        return z_inv

    @torch.no_grad()
    def extract_full_factorized_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts both z_inv [B, d_inv] and z_meas [B, d_meas] for downstream auditing.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        tokens, pos = self.tokenizer(x)
        H = self.context_encoder(tokens, pos)
        h_center = H[:, 0, :]
        z_inv, z_meas = self.factorized_head(h_center)
        return z_inv, z_meas
