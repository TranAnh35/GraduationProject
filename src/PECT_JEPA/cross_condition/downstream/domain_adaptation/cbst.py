"""
Class-Balanced Self-Training (CBST) with Prototype Pseudo-Labeling.

Non-alignment / Self-Training domain adaptation:
Instead of forcing marginal feature distributions to overlap (which suffers from severe
negative transfer under extreme label shift), CBST iteratively generates high-confidence
pseudo-labels for the target domain while enforcing a class-balanced target prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CBSTModel(nn.Module):
    """
    Class-Balanced Self-Training Network:
    - Feature Adapter: MLP mapping f(z)
    - Classifier: Linear head g(f(z)) -> logit
    - Prototype bank: Defect prototype p_1 and Sound prototype p_0
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

        # Running prototypes for defect (1) and sound metal (0)
        self.register_buffer("proto_0", torch.zeros(hidden_dim))
        self.register_buffer("proto_1", torch.zeros(hidden_dim))
        self.register_buffer("proto_initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_net(x)
        logits = self.classifier(feat)
        return logits.squeeze(-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

    def update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor, momentum: float = 0.9):
        """
        Updates running source prototypes for sound (0) and defect (1).
        """
        with torch.no_grad():
            mask_0 = (labels == 0)
            mask_1 = (labels == 1)
            if mask_0.sum() > 0:
                mean_0 = F.normalize(feats[mask_0].mean(dim=0), dim=-1)
                if not self.proto_initialized:
                    self.proto_0.copy_(mean_0)
                else:
                    self.proto_0.copy_(momentum * self.proto_0 + (1 - momentum) * mean_0)

            if mask_1.sum() > 0:
                mean_1 = F.normalize(feats[mask_1].mean(dim=0), dim=-1)
                if not self.proto_initialized:
                    self.proto_1.copy_(mean_1)
                else:
                    self.proto_1.copy_(momentum * self.proto_1 + (1 - momentum) * mean_1)

            if mask_0.sum() > 0 and mask_1.sum() > 0:
                self.proto_initialized.copy_(torch.tensor(True))

    def generate_balanced_pseudo_labels(
        self,
        target_feats: torch.Tensor,
        top_k_defect: int = 16,
        top_k_sound: int = 64,
        temperature: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates class-balanced pseudo-labels for target samples using prototype distance.
        Guarantees that defect candidates are NOT washed out by majority sound metal!
        
        Returns:
            selected_feats: [N_selected, hidden_dim]
            pseudo_labels:  [N_selected] (0 or 1)
        """
        if not self.proto_initialized:
            return torch.empty(0, target_feats.shape[1], device=target_feats.device), torch.empty(0, device=target_feats.device)

        norm_feats = F.normalize(target_feats, dim=-1)
        sim_0 = torch.matmul(norm_feats, F.normalize(self.proto_0, dim=-1)) / temperature
        sim_1 = torch.matmul(norm_feats, F.normalize(self.proto_1, dim=-1)) / temperature

        # Relative margin / defect score
        defect_score = sim_1 - sim_0

        B = target_feats.shape[0]
        k_def = min(top_k_defect, B // 4)
        k_snd = min(top_k_sound, B - k_def)

        if k_def <= 0 or k_snd <= 0:
            return torch.empty(0, target_feats.shape[1], device=target_feats.device), torch.empty(0, device=target_feats.device)

        # Select top k_def with highest defect score
        _, idx_def = torch.topk(defect_score, k=k_def, largest=True)
        # Select top k_snd with lowest defect score (highest sound confidence)
        _, idx_snd = torch.topk(defect_score, k=k_snd, largest=False)

        sel_idx = torch.cat([idx_def, idx_snd])
        pseudo_y = torch.cat([torch.ones(k_def, device=target_feats.device),
                              torch.zeros(k_snd, device=target_feats.device)])

        return target_feats[sel_idx], pseudo_y

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192, device: str = "cpu") -> np.ndarray:
        self.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                bx = torch.from_numpy(x[i:i+batch_size]).float().to(device)
                logits = self(bx)
                prob = torch.sigmoid(logits).cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs, axis=0) if probs else np.array([])
