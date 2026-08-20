"""
Configuration definitions for PECT-JEPA.
Implements the configurable parameter structure defined in Section 20 of PECT_JEPA_Implementation_Spec_v0.1.md.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any
import json
import os


@dataclass
class DataConfig:
    data_dir: str = "data"
    time_samples: int = 500
    normalization: str = "min_max"  # 'min_max', 'standard', 'none'
    raster_correction: bool = True
    spatial_size: Optional[Tuple[int, int]] = (300, 300)  # None or (H, W) to crop/pad uniformly
    cache_in_memory: bool = False
    eps: float = 1e-8


@dataclass
class TemporalEncoderConfig:
    t_prime: int = 64  # Latent temporal positions: candidate 64 or 128
    raw_samples: int = 500
    conv_branches: int = 3
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [5, 9, 15])
    conv_dilations: List[int] = field(default_factory=lambda: [1, 1, 2])
    hidden_dim: int = 32
    transformer_blocks: int = 2
    transformer_heads: int = 4
    dropout: float = 0.0


@dataclass
class ClipConfig:
    temporal_length: int = 16  # Tc = 16
    stride: int = 8  # clip_stride: 8, 12, or 16


@dataclass
class TokenizerConfig:
    spatial_patch: int = 8  # Ps: candidate 4, 8, or 16
    embed_dim: int = 128  # D: token embedding dimension
    pos_embed_type: str = "learnable"  # 'learnable', 'sinusoidal'
    dropout: float = 0.0


@dataclass
class MaskConfig:
    mask_type: str = "spatiotemporal_block"
    spatial_block_h: int = 4  # Bh: tokens in height
    spatial_block_w: int = 4  # Bw: tokens in width
    temporal_block_t: int = 1  # Bt: clips in temporal axis
    num_blocks: int = 4  # Number of target blocks sampled per sample
    min_mask_ratio: float = 0.2
    max_mask_ratio: float = 0.6
    dynamic: bool = True


@dataclass
class EncoderConfig:
    attention_type: str = "factorized"  # 'factorized', 'window', 'full'
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    window_size: Tuple[int, int] = (4, 4)  # For window attention


@dataclass
class TargetEncoderConfig:
    ema_momentum: float = 0.996  # Starting EMA momentum m
    ema_momentum_end: float = 1.0  # Optional cosine scheduler end momentum
    use_momentum_schedule: bool = True


@dataclass
class PredictorConfig:
    depth: int = 2  # Lightweight predictor: 2-4 Transformer blocks
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class LossConfig:
    loss_type: str = "normalized_l2"  # 'normalized_l2', 'cosine', 'smooth_l1'
    eps: float = 1e-8


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 2
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    epochs: int = 50
    grad_clip: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda"  # 'cuda' or 'cpu'
    save_dir: str = "checkpoints/pect_jepa"
    log_interval: int = 10
    val_interval: int = 1


@dataclass
class PECTJEPAConfig:
    data: DataConfig = field(default_factory=DataConfig)
    temporal_encoder: TemporalEncoderConfig = field(default_factory=TemporalEncoderConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    target_encoder: TargetEncoderConfig = field(default_factory=TargetEncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PECTJEPAConfig":
        return cls(
            data=DataConfig(**data.get("data", {})),
            temporal_encoder=TemporalEncoderConfig(**data.get("temporal_encoder", {})),
            clip=ClipConfig(**data.get("clip", {})),
            tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
            mask=MaskConfig(**data.get("mask", {})),
            encoder=EncoderConfig(**data.get("encoder", {})),
            target_encoder=TargetEncoderConfig(**data.get("target_encoder", {})),
            predictor=PredictorConfig(**data.get("predictor", {})),
            loss=LossConfig(**data.get("loss", {})),
            training=TrainingConfig(**data.get("training", {})),
        )

    @classmethod
    def from_json(cls, file_path: str) -> "PECTJEPAConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_config() -> PECTJEPAConfig:
    """Returns default PECT-JEPA v0.1 configuration."""
    return PECTJEPAConfig()
