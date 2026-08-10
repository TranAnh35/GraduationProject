"""
Configuration for HeSpatial-DSBN
Heterogeneous Spatial Context Autoencoder with Domain-Specific Batch Normalization
for Cross-Sensor and Cross-Material PECT Domain Adaptation.

Standalone configuration file.
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple

# =============================================================================
# SPATIAL CONTEXT PARAMETERS
# =============================================================================
RADIUS = 5
DILATION = 1
SCALES: List[Tuple[int, int]] = [(RADIUS, DILATION)]
NUM_DIRECTIONS = 8                  # 4 lines through center (Vertical, Horizontal, Main & Anti Diagonals)
NUM_CONTEXT_CHANNELS = 8 * RADIUS  # 40 channels for r=5
TIME_SAMPLES = 500                  # T = 500 time points

# =============================================================================
# MODEL ARCHITECTURE PARAMETERS
# =============================================================================
FILTERS_START = 16
LATENT_DIM = 32
K_SPARSE_RATIO = 0.25                # Keep top-k = max(1, LATENT_DIM * K_SPARSE_RATIO) = 8
DROPOUT_RATE = 0.2
ACTIVATION = "swish"                # Swish activation across all conv layers

# Sensor Adapter settings
ADAPTER_PROJ_CHANNELS = 16          # Projection depth for Sensor-Specific Adapters
SENSOR_TYPES = ["hall", "coil", "diffensor"]
MATERIAL_TYPES = ["al", "steel"]

# Total domains (3 Sensors x 2 Materials = 6 combined domains)
NUM_DOMAINS = len(SENSOR_TYPES) * len(MATERIAL_TYPES)
DOMAIN_MAP: Dict[Tuple[str, str], int] = {
    ("hall", "al"): 0,
    ("hall", "steel"): 1,
    ("coil", "al"): 2,
    ("coil", "steel"): 3,
    ("diffensor", "al"): 4,
    ("diffensor", "steel"): 5,
}

# =============================================================================
# TRAINING & HYPERPARAMETERS
# =============================================================================
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 5e-5
PATIENCE = 7

# Loss Weights
LAMBDA_REC_S = 1.0                  # Source reconstruction weight
LAMBDA_REC_T = 0.2                  # Target pseudo-normal reconstruction weight
LAMBDA_MMD = 0.1                   # Maximum Mean Discrepancy alignment weight
LAMBDA_GRL = 0.05                  # Gradient Reversal Layer adversarial weight
LAMBDA_SVDD = 0.01                 # SVDD hypersphere compactness weight

# 2-Stage Training Schedule
WARMUP_EPOCHS = 5                   # Stage 1: Warm-up on Source domain only
RAMP_EPOCHS = 15                    # Stage 2: Ramp-up GRL & Pseudo-Normal selection
K_START_RATIO = 0.5                # Initial pseudo-normal quantile (50%)
K_END_RATIO = 0.8                  # Final pseudo-normal quantile (80%)

# Alignment parameters
MMD_MAX_SAMPLES = 128
DSBN_ENABLED = True
DOMAIN_HIDDEN = 64                  # Hidden units in GRL Domain Head

# =============================================================================
# 6 CROSS-SENSOR BENCHMARK TRANSFER TASKS
# =============================================================================
TRANSFER_TASKS: Dict[str, Dict[str, str]] = {
    "Task_1": {"src_sensor": "hall", "src_mat": "al", "tgt_sensor": "coil", "tgt_mat": "steel"},
    "Task_2": {"src_sensor": "coil", "src_mat": "steel", "tgt_sensor": "hall", "tgt_mat": "al"},
    "Task_3": {"src_sensor": "hall", "src_mat": "al", "tgt_sensor": "diffensor", "tgt_mat": "steel"},
    "Task_4": {"src_sensor": "diffensor", "src_mat": "steel", "tgt_sensor": "hall", "tgt_mat": "al"},
    "Task_5": {"src_sensor": "coil", "src_mat": "al", "tgt_sensor": "diffensor", "tgt_mat": "steel"},
    "Task_6": {"src_sensor": "diffensor", "src_mat": "steel", "tgt_sensor": "coil", "tgt_mat": "al"},
}

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for directory in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
