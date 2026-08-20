"""
Loss Functions & Alignment Modules for HeSpatial-DSBN.
Includes GRL (Gradient Reversal Layer), Multi-Scale MMD, Target Pseudo-Normal Quantile Selection,
SVDD Hypersphere Compactness Loss, and Multi-Class Domain Adversarial Loss.

Standalone module - no external dependencies.
"""

from __future__ import annotations
from typing import Dict, Tuple
import tensorflow as tf


# =============================================================================
# GRADIENT REVERSAL LAYER (GRL)
# =============================================================================

@tf.custom_gradient
def grad_reverse(x: tf.Tensor, lambda_val: tf.Tensor):
    """Reverses gradient sign during backward pass: dL/dx = -lambda * dy."""
    def custom_grad(dy):
        return -lambda_val * dy, None
    return tf.identity(x), custom_grad


class GradientReversalLayer(tf.keras.layers.Layer):
    """Gradient Reversal Layer (GRL) for domain adversarial training with dynamic lambda schedule."""
    def __init__(self, lambda_value: float = 1.0, name: str = "grl", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_val = tf.Variable(
            initial_value=float(lambda_value),
            trainable=False,
            dtype=tf.float32,
            name=f"{name}_lambda"
        )

    def set_lambda(self, val: float):
        """Dynamically update GRL reversal strength."""
        self.lambda_val.assign(float(val))

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        if training:
            return grad_reverse(x, self.lambda_val)
        return tf.identity(x)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"lambda_value": float(self.lambda_val.numpy()) if hasattr(self.lambda_val, 'numpy') else 1.0})
        return cfg


# =============================================================================
# RECONSTRUCTION & PSEUDO-NORMAL SELECTION LOSSES
# =============================================================================

def mae_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute Mean Absolute Error (MAE)."""
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))


def mae_per_sample(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute MAE per sample across time and channel dimensions."""
    diff = tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))
    return tf.reduce_mean(diff, axis=list(range(1, len(diff.shape))))


def select_pseudo_normal(
    y_true: tf.Tensor, y_pred: tf.Tensor, k_ratio: float = 0.5
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Online selection of Pseudo-Normal target samples based on lowest reconstruction error.
    Returns (pseudo_y_true, pseudo_y_pred, pseudo_indices).
    """
    errors = mae_per_sample(y_true, y_pred)
    batch_size = tf.cast(tf.shape(errors)[0], tf.float32)
    k = tf.maximum(1, tf.cast(batch_size * k_ratio, tf.int32))
    _, indices = tf.math.top_k(-errors, k=k)
    return tf.gather(y_true, indices), tf.gather(y_pred, indices), indices


# =============================================================================
# SVDD HYPERSPHERE COMPACTNESS LOSS
# =============================================================================

def compute_svdd_center(z: tf.Tensor) -> tf.Tensor:
    """Compute hypersphere center c in latent space for normal samples."""
    if len(z.shape) == 3:
        z = tf.reduce_mean(z, axis=1)  # Temporal pooling
    return tf.reduce_mean(z, axis=0, keepdims=True)


def svdd_compactness_loss(z: tf.Tensor, center: tf.Tensor) -> tf.Tensor:
    """SVDD loss: ||z - c||^2."""
    if len(z.shape) == 3:
        z = tf.reduce_mean(z, axis=1)
    dist_sq = tf.reduce_sum(tf.square(z - center), axis=-1)
    return tf.reduce_mean(dist_sq)


# =============================================================================
# MAXIMUM MEAN DISCREPANCY (MMD) ALIGNMENT
# =============================================================================

def _gaussian_kernel_matrix(x: tf.Tensor, y: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
    x_exp = tf.expand_dims(tf.cast(x, tf.float32), axis=1)
    y_exp = tf.expand_dims(tf.cast(y, tf.float32), axis=0)
    dist_sq = tf.reduce_sum(tf.square(x_exp - y_exp), axis=2)
    beta = 1.0 / (2.0 * tf.reshape(sigmas, (-1, 1, 1)))
    return tf.reduce_sum(tf.exp(-beta * tf.expand_dims(dist_sq, axis=0)), axis=0)


def maximum_mean_discrepancy(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Multi-scale Gaussian Kernel MMD distance."""
    sigmas = tf.constant([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=tf.float32)
    k_xx = _gaussian_kernel_matrix(x, x, sigmas)
    k_yy = _gaussian_kernel_matrix(y, y, sigmas)
    k_xy = _gaussian_kernel_matrix(x, y, sigmas)
    
    n = tf.cast(tf.shape(x)[0], tf.float32)
    m = tf.cast(tf.shape(y)[0], tf.float32)
    
    sum_k_xx = (tf.reduce_sum(k_xx) - tf.reduce_sum(tf.linalg.diag_part(k_xx))) / (n * (n - 1.0) + 1e-8)
    sum_k_yy = (tf.reduce_sum(k_yy) - tf.reduce_sum(tf.linalg.diag_part(k_yy))) / (m * (m - 1.0) + 1e-8)
    sum_k_xy = tf.reduce_mean(k_xy)
    
    return tf.maximum(sum_k_xx + sum_k_yy - 2.0 * sum_k_xy, 0.0)


def mmd_loss(z_s: tf.Tensor, z_t: tf.Tensor, max_samples: int = 128) -> tf.Tensor:
    """Subsampled MMD loss between source and target latent feature maps."""
    if len(z_s.shape) == 3: z_s = tf.reduce_mean(z_s, axis=1)
    if len(z_t.shape) == 3: z_t = tf.reduce_mean(z_t, axis=1)
    
    n_s, n_t = tf.shape(z_s)[0], tf.shape(z_t)[0]
    idx_s = tf.random.shuffle(tf.range(n_s))[:tf.minimum(n_s, max_samples)]
    idx_t = tf.random.shuffle(tf.range(n_t))[:tf.minimum(n_t, max_samples)]
    
    return maximum_mean_discrepancy(tf.gather(z_s, idx_s), tf.gather(z_t, idx_t))


# =============================================================================
# COMBINED HESPATIAL-DSBN MULTI-OBJECTIVE LOSS CLASS
# =============================================================================

class HeSpatialDSBNLoss:
    """
    Combined Loss for HeSpatial-DSBN:
    L_total = L_rec_S + alpha_T * L_rec_T,pn + lambda_svdd * L_svdd + lambda_mmd * L_mmd + lambda_grl * L_adv
    """
    def __init__(
        self,
        lambda_mmd: float = 0.1,
        lambda_grl: float = 0.05,
        lambda_svdd: float = 0.01,
        mmd_max_samples: int = 128,
    ):
        self.lambda_mmd = lambda_mmd
        self.lambda_grl = lambda_grl
        self.lambda_svdd = lambda_svdd
        self.mmd_max_samples = mmd_max_samples

    def __call__(
        self,
        y_s_true: tf.Tensor,
        y_s_pred: tf.Tensor,
        y_t_true: tf.Tensor,
        y_t_pred: tf.Tensor,
        z_source: tf.Tensor,
        z_target: tf.Tensor,
        domain_logits: tf.Tensor,
        domain_labels: tf.Tensor,
        alpha_t: float = 0.2,
        k_ratio: float = 0.5,
        svdd_center: tf.Tensor = None,
        current_lambda_grl: float = None,
    ) -> Dict[str, tf.Tensor]:
        
        # 1. Source Reconstruction Loss
        loss_rec_s = mae_loss(y_s_true, y_s_pred)

        # 2. Target Pseudo-Normal Reconstruction Loss
        _, _, pseudo_indices = select_pseudo_normal(y_t_true, y_t_pred, k_ratio)
        if k_ratio > 0 and alpha_t > 0:
            pyt_t = tf.gather(y_t_true, pseudo_indices)
            pyt_p = tf.gather(y_t_pred, pseudo_indices)
            loss_rec_t = mae_loss(pyt_t, pyt_p)
        else:
            loss_rec_t = tf.constant(0.0)

        # 3. SVDD Hypersphere Compactness Loss
        if svdd_center is None:
            svdd_center = compute_svdd_center(z_source)
        loss_svdd = svdd_compactness_loss(z_source, svdd_center)

        # 4. Selective MMD Alignment Loss
        z_target_aligned = tf.gather(z_target, pseudo_indices)
        loss_mmd = mmd_loss(z_source, z_target_aligned, self.mmd_max_samples)

        # 5. Domain Classifier Adversarial Loss (Multi-Class or Binary)
        if domain_logits is not None and domain_labels is not None:
            if domain_logits.shape[-1] == 1 or len(domain_logits.shape) == 1:
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                loss_adv = bce(tf.reshape(domain_labels, [-1]), tf.cast(tf.reshape(domain_logits, [-1]), tf.float32))
            else:
                cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                loss_adv = cce(domain_labels, domain_logits)
        else:
            loss_adv = tf.constant(0.0)

        # Total Weighted Loss
        w_grl = current_lambda_grl if current_lambda_grl is not None else self.lambda_grl
        total_loss = (
            loss_rec_s
            + alpha_t * loss_rec_t
            + self.lambda_svdd * loss_svdd
            + self.lambda_mmd * loss_mmd
            + w_grl * loss_adv
        )

        return {
            "total": total_loss,
            "rec_s": loss_rec_s,
            "rec_t": loss_rec_t,
            "svdd": loss_svdd,
            "mmd": loss_mmd,
            "adv": loss_adv,
        }
