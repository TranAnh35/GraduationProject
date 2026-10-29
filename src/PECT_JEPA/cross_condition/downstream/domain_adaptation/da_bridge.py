"""
Domain Adaptation Bridge Suite for Work Order WO-20261029-DA.

Implements matched-control Domain Adaptation algorithms on frozen PECT representations:
- T1a: Closed-form CORAL (Sun et al. AAAI 2016) with source-to-target covariance alignment.
- T1b: DANN N0 (lambda=0 matched neural baseline) and N1 (lambda>0 adversarial alignment)
       with BatchNorm-free residual adapter and strictly isolated task/domain samplers.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# -----------------------------------------------------------------------------
# T1a: Closed-Form CORAL (Sun et al. AAAI 2016)
# -----------------------------------------------------------------------------

class ClosedFormCORAL:
    """
    Closed-form Correlation Alignment (CORAL) for feature representations.

    Transforms source features so their covariance matches target covariance:
    C_s = Cov(X_s) + alpha_s * I
    C_t = Cov(X_t) + alpha_t * I
    A = C_s^{-1/2} @ C_t^{1/2}
    X_s^* = (X_s - mu_s) @ A + mu_t

    Target features are evaluated directly (identity) in target space.
    """
    def __init__(self, ridge: float = 1e-2):
        self.ridge = float(ridge)
        self.mu_s: Optional[np.ndarray] = None
        self.mu_t: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.C_s: Optional[np.ndarray] = None
        self.C_t: Optional[np.ndarray] = None
        self.alpha_s: float = 0.0
        self.alpha_t: float = 0.0

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "ClosedFormCORAL":
        Xs = np.asarray(source_features, dtype=np.float64)
        Xt = np.asarray(target_features, dtype=np.float64)
        ns, d = Xs.shape
        nt, dt = Xt.shape
        if d != dt:
            raise ValueError(f"Dimension mismatch: source {d} vs target {dt}")
        if ns < 2 or nt < 2:
            raise ValueError(f"Need at least 2 samples: ns={ns}, nt={nt}")

        self.mu_s = np.mean(Xs, axis=0, keepdims=True)
        self.mu_t = np.mean(Xt, axis=0, keepdims=True)

        Xs_c = Xs - self.mu_s
        Xt_c = Xt - self.mu_t

        self.C_s = (Xs_c.T @ Xs_c) / (ns - 1)
        self.C_t = (Xt_c.T @ Xt_c) / (nt - 1)

        # Ridge loading proportional to trace / dimension
        tr_s = float(np.trace(self.C_s))
        tr_t = float(np.trace(self.C_t))
        self.alpha_s = self.ridge * max(tr_s / d, 1e-6)
        self.alpha_t = self.ridge * max(tr_t / d, 1e-6)

        Cs_reg = self.C_s + self.alpha_s * np.eye(d, dtype=np.float64)
        Ct_reg = self.C_t + self.alpha_t * np.eye(d, dtype=np.float64)

        # Symmetric eigendecomposition
        ws, vs = np.linalg.eigh(Cs_reg)
        ws = np.maximum(ws, 1e-12)
        inv_sqrt_Cs = vs @ np.diag(1.0 / np.sqrt(ws)) @ vs.T

        wt, vt = np.linalg.eigh(Ct_reg)
        wt = np.maximum(wt, 1e-12)
        sqrt_Ct = vt @ np.diag(np.sqrt(wt)) @ vt.T

        self.A = inv_sqrt_Cs @ sqrt_Ct
        return self

    def transform_source(self, source_features: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise RuntimeError("ClosedFormCORAL must be fitted before transform.")
        Xs = np.asarray(source_features, dtype=np.float64)
        Xs_c = Xs - self.mu_s
        Xs_aligned = Xs_c @ self.A + self.mu_t
        return Xs_aligned.astype(np.float32)

    def transform_target(self, target_features: np.ndarray) -> np.ndarray:
        # Target space is the reference destination
        return np.asarray(target_features, dtype=np.float32)


# -----------------------------------------------------------------------------
# T1b: DANN N0 / N1 Architecture and Modules
# -----------------------------------------------------------------------------

class ReverseGrad(Function):
    """
    Gradient Reversal Layer (GRL).
    Forward: identity.
    Backward: scales gradient by -alpha.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return ReverseGrad.apply(x, alpha)


class DANNResidualModule(nn.Module):
    """
    BatchNorm-free Residual Adapter, Classifier, and Domain Discriminator.
    Eliminates BatchNorm running statistics confounds (FM23).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # 1. Residual Feature Adapter: g_phi(z) = z + MLP(LN(z))
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )
        # Zero initialization on final adapter layer: initially g_phi(z) == z
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        # 2. Defect Task Classifier: LayerNorm, NO BatchNorm
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 3. Domain Discriminator: LayerNorm, NO BatchNorm
        self.discriminator = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        z_adapted = self.adapt(x)
        class_logits = self.classifier(z_adapted).squeeze(-1)
        z_rev = grad_reverse(z_adapted, alpha=alpha)
        domain_logits = self.discriminator(z_rev).squeeze(-1)
        return class_logits, domain_logits

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        z_adapted = self.adapt(x)
        return self.classifier(z_adapted).squeeze(-1)

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192, device: torch.device = torch.device("cpu")) -> np.ndarray:
        self.eval()
        self.to(device)
        probs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                bx = torch.from_numpy(x[i : i + batch_size]).float().to(device)
                logits = self.predict_logits(bx)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float32)


# -----------------------------------------------------------------------------
# Training and Sampling Pipeline for N0 and N1
# -----------------------------------------------------------------------------

class ClassBalancedSourceSampler:
    """Samples batches from source with 50/50 defect / background balance."""
    def __init__(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState):
        self.X = X
        self.y = y
        self.rng = rng
        self.pos_idx = np.where(y == 1)[0]
        self.neg_idx = np.where(y == 0)[0]
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError(f"Need both classes: pos={len(self.pos_idx)}, neg={len(self.neg_idx)}")

    def sample_batch(self, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos
        replace_pos = len(self.pos_idx) < n_pos
        replace_neg = len(self.neg_idx) < n_neg
        sel_pos = self.rng.choice(self.pos_idx, size=n_pos, replace=replace_pos)
        sel_neg = self.rng.choice(self.neg_idx, size=n_neg, replace=replace_neg)
        sel = np.concatenate([sel_pos, sel_neg])
        self.rng.shuffle(sel)
        return self.X[sel], self.y[sel]


class NaturalDomainSampler:
    """Samples batches from marginal natural distribution without GT awareness."""
    def __init__(self, X: np.ndarray, rng: np.random.RandomState):
        self.X = X
        self.rng = rng
        self.n = len(X)
        if self.n == 0:
            raise ValueError("Feature array cannot be empty")

    def sample_batch(self, batch_size: int = 256) -> np.ndarray:
        replace = self.n < batch_size
        sel = self.rng.choice(self.n, size=batch_size, replace=replace)
        return self.X[sel]


def train_dann_model(
    source_task_X: np.ndarray,
    source_task_y: np.ndarray,
    source_domain_X: np.ndarray,
    target_domain_X: np.ndarray,
    in_dim: int,
    lambda_align: float,
    seed: int = 42,
    warmup_steps: int = 200,
    adapt_steps: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DANNResidualModule, Dict[str, Any]]:
    """
    Trains DANNResidualModule for N0 (lambda_align=0) or N1 (lambda_align>0).

    Schedule:
    - Step 1..warmup_steps: Source task pre-training (lambda=0).
    - Step warmup_steps+1..warmup_steps+adapt_steps: Continued training with lambda_align.
    """
    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed)

    model = DANNResidualModule(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    task_criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    task_sampler = ClassBalancedSourceSampler(source_task_X, source_task_y, np.random.RandomState(seed + 1))
    src_dom_sampler = NaturalDomainSampler(source_domain_X, np.random.RandomState(seed + 2))
    tgt_dom_sampler = NaturalDomainSampler(target_domain_X, np.random.RandomState(seed + 3))

    total_steps = warmup_steps + adapt_steps
    logs: List[Dict[str, float]] = []

    model.train()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad()

        # 1. Task loss on class-balanced source batch
        bx_task, by_task = task_sampler.sample_batch(batch_size=batch_size)
        t_x = torch.from_numpy(bx_task).float().to(device)
        t_y = torch.from_numpy(by_task).float().to(device)

        logits_task = model.predict_logits(t_x)
        loss_task = task_criterion(logits_task, t_y)

        # 2. Domain loss
        effective_lambda = 0.0
        loss_domain = torch.tensor(0.0, device=device)
        grad_norm_adapter = 0.0

        if step > warmup_steps and lambda_align > 0.0:
            effective_lambda = lambda_align
            # Standard Ganin warmup factor in [0, 1]
            p = float(step - warmup_steps) / float(adapt_steps)
            alpha_grl = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

            bx_src = src_dom_sampler.sample_batch(batch_size=batch_size)
            bx_tgt = tgt_dom_sampler.sample_batch(batch_size=batch_size)

            d_src_x = torch.from_numpy(bx_src).float().to(device)
            d_tgt_x = torch.from_numpy(bx_tgt).float().to(device)

            # Source domain label = 0, Target domain label = 1
            d_x = torch.cat([d_src_x, d_tgt_x], dim=0)
            d_y = torch.cat([
                torch.zeros(len(bx_src), device=device),
                torch.ones(len(bx_tgt), device=device),
            ], dim=0)

            _, dom_logits = model(d_x, alpha=alpha_grl)
            loss_domain = domain_criterion(dom_logits, d_y)
            total_loss = loss_task + effective_lambda * loss_domain
        else:
            total_loss = loss_task

        total_loss.backward()

        # Measure adapter gradient norm
        adapter_grads = [p.grad for p in model.adapter.parameters() if p.grad is not None]
        if adapter_grads:
            grad_norm_adapter = float(torch.norm(torch.stack([torch.norm(g) for g in adapter_grads])).item())

        optimizer.step()

        if step % 50 == 0 or step == total_steps:
            logs.append({
                "step": step,
                "loss_task": float(loss_task.item()),
                "loss_domain": float(loss_domain.item()),
                "effective_lambda": float(effective_lambda),
                "adapter_grad_norm": float(grad_norm_adapter),
            })

    model.eval()
    meta = {
        "lambda_align": lambda_align,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "adapt_steps": adapt_steps,
        "batch_size": batch_size,
        "training_logs": logs,
        "final_loss_task": logs[-1]["loss_task"] if logs else None,
        "final_loss_domain": logs[-1]["loss_domain"] if logs else None,
        "final_adapter_grad_norm": logs[-1]["adapter_grad_norm"] if logs else None,
    }
    return model, meta
