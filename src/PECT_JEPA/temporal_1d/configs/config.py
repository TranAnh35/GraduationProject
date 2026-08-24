"""
Configuration definitions for 1D Temporal PECT-JEPA (TS-JEPA) (implement.md, Section 4.1).
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import os


@dataclass
class Temporal1DConfig:
    # Data
    data_dir: str = "data"
    time_samples: int = 500        # Số điểm thời gian raw (500)
    normalization: str = "min_max" # Chuẩn hóa Min-Max per-file [0, 1]
    use_memmap: bool = True
    cache_dir: str = ".cache/pect_1d_mmap"
    eps: float = 1e-8
    
    # Tokenizer 1D
    patch_length: int = 32         # Độ dài mỗi patch thời gian (P = 32)
    stride: int = 31               # Stride để chia 500 điểm thành N=16 patches
    num_patches: int = 16          # Tổng số patches (N = 16)
    embed_dim: int = 128           # Chiều vector nhúng D = 128
    pos_embed_type: str = "sinusoidal" # 'sinusoidal' hoặc 'learnable'
    dropout: float = 0.0
    
    # Masking 1D
    mask_strategy: str = "late_decay"  # 'late_decay' (vật lý) hoặc 'random_patch'
    mask_ratio: float = 0.70       # Tỷ lệ che ~70% (11/16 patches bị che)
    num_visible_early: int = 5     # Số patch đầu được mở trong late_decay
    
    # Architecture
    encoder_depth: int = 4         # 4 Transformer Blocks cho Context Encoder
    encoder_heads: int = 4
    predictor_depth: int = 2       # 2 Transformer Blocks cho Predictor
    predictor_heads: int = 4
    mlp_ratio: float = 4.0
    ema_momentum: float = 0.996
    ema_momentum_end: float = 1.0
    use_momentum_schedule: bool = True
    
    # Loss
    loss_type: str = "normalized_l2"  # 'normalized_l2', 'cosine', 'smooth_l1'
    
    # Training
    batch_size: int = 512          # Batch size lớn cho mô hình 1D
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    epochs: int = 50
    grad_clip: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda"
    save_dir: str = "checkpoints/pect_jepa_1d"
    log_interval: int = 20
    val_interval: int = 1
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Temporal1DConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, file_path: str) -> "Temporal1DConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_config_1d() -> Temporal1DConfig:
    """Returns default 1D Temporal PECT-JEPA configuration."""
    return Temporal1DConfig()
