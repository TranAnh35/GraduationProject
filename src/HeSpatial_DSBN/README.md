# HeSpatial-DSBN: Heterogeneous Spatial Context Autoencoder with Domain-Specific Batch Normalization

Standalone implementation of the proposed **HeSpatial-DSBN** framework for Cross-Sensor and Cross-Material PECT (Pulsed Eddy Current Testing) Domain Adaptation & Anomaly Detection.

---

## 📌 Architecture Overview

HeSpatial-DSBN addresses three challenges simultaneously:
1. **Unsupervised Anomaly Detection**: Spatial Context Autoencoder reconstructing center waveform $y_{i,j}$ from 4-line 8-direction neighborhood $X \in \mathbb{R}^{T \times 8r}$.
2. **Cross-Material Domain Adaptation**: Domain-Specific Batch Normalization (DSBN) isolating amplitude stats $(\gamma_{s,m}, \beta_{s,m})$ for Aluminum 2024 (diffusion) and Carbon Steel C45 (MFL).
3. **Cross-Sensor Domain Adaptation**: Heterogeneous Sensor-Specific Adapters ($g_{\phi_{\text{Hall}}}$, $g_{\phi_{\text{Coil}}}$, $g_{\phi_{\text{Diff}}}$) mapping $B(t)$, $dB(t)/dt$, and $\Delta B(t)$ to a unified feature representation.

```
Inputs (Hall, Coil, Diffensor)
     │
     ▼
[Sensor-Specific Adapters] ──> Projection to unified channel space
     │
     ▼
[Shared Spatial Engine Core] ──> Residual Separable Conv1D (Swish) + GRU Bottleneck + K-Sparse Layer
     │
     ▼
[Multi-Domain DSBN] ──────────> Domain-specific affine parameters (γ_sm, β_sm) for 6 domains
     │
     ├───────────────► [GRL Domain Discriminator] (Sensor & Material Adversarial Alignment)
     ▼
[Transposed Conv Decoder] ───> Center Trace Reconstruction ŷ_ij ∈ R^T
```

---

## 📁 Package Directory Structure

```
source/HeSpatial_DSBN/
├── __init__.py           # Package initializer
├── config.py             # Hyperparameters, domain mapping, 6 transfer tasks
├── dataloader.py         # StarContextSampler & MultiDomainPECTDataset
├── processing.py         # Normalization, SVD denoise, CNR, NCC, SSIM, ROC-AUC
├── losses.py             # GRL, MMD, Target Pseudo-Normal Selection, SVDD Loss
├── model.py              # MultiDomainBatchNorm, SensorAdapter, Encoder, Decoder, Discriminator
├── train.py              # Two-stage training script (Warmup + Adversarial Rampup)
├── evaluate.py           # Benchmark test runner across 6 transfer tasks
└── README.md             # Package documentation
```

---

## 🚀 Usage Guide

### 1. Training a Transfer Task
Run 2-Stage training on Task 1 (Hall/Aluminum $\rightarrow$ Coil/Steel):
```bash
python -m source.HeSpatial_DSBN.train --task Task_1 --epochs 30 --batch_size 64
```

Supported task names: `Task_1`, `Task_2`, `Task_3`, `Task_4`, `Task_5`, `Task_6`.

### 2. Evaluating Benchmark Results
Run evaluation across all 6 Cross-Sensor Transfer Tasks:
```bash
python -m source.HeSpatial_DSBN.evaluate --task all
```

---

## 📊 6 Cross-Sensor Transfer Tasks Matrix

| Task Code | Source Domain ($S$) | Target Domain ($T$) | Topology Shift |
|---|---|---|---|
| **Task_1** | Hall ($B$, Alu) | Coil ($dB/dt$, Steel) | Absolute Field $\rightarrow$ Time Derivative |
| **Task_2** | Coil ($dB/dt$, Steel) | Hall ($B$, Alu) | Time Derivative $\rightarrow$ Absolute Field |
| **Task_3** | Hall ($B$, Alu) | Diffensor ($\Delta B$, Steel) | Absolute Field $\rightarrow$ Spatial Differential |
| **Task_4** | Diffensor ($\Delta B$, Steel) | Hall ($B$, Alu) | Spatial Differential $\rightarrow$ Absolute Field |
| **Task_5** | Coil ($dB/dt$, Alu) | Diffensor ($\Delta B$, Steel) | Time Derivative $\rightarrow$ Spatial Differential |
| **Task_6** | Diffensor ($\Delta B$, Steel) | Coil ($dB/dt$, Alu) | Spatial Differential $\rightarrow$ Time Derivative |

---

## ⚙️ Key Hyperparameters

* **Spatial Neighborhood**: Radius $r=5$, Dilation $d=1$ ($8r = 40$ context channels).
* **Network Layers**: 3 Residual Separable Conv1D blocks with **Swish** activation ($T \rightarrow T/4$).
* **Recurrent Bottleneck**: GRU ($32$ units) + $k$-Sparse layer ($k=16$).
* **Loss Objective**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}}^{S} + \alpha_T \mathcal{L}_{\text{rec}}^{T,pn} + \lambda_{\text{svdd}} \mathcal{L}_{\text{svdd}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{mmd}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$.
