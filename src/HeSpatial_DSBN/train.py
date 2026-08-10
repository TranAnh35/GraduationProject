"""
Two-Stage Training Script for HeSpatial-DSBN.
Implements Warm-up Stage, GRL Adversarial Adaptation Stage, Target Pseudo-Normal Ramp-Up,
and Multi-Domain DSBN Optimization across Cross-Sensor Transfer Tasks.

Standalone executable script - no external project dependencies.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
import tensorflow as tf

from . import config
from .dataloader import StarContextSampler, get_default_normal_ranges
from .losses import HeSpatialDSBNLoss, compute_svdd_center
from .model import build_hespatial_dsbn_model, build_inference_model
from .processing import evaluate_reconstruction_metrics


def set_seed(seed: int = 42):
    """Set random seeds for numpy and tensorflow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_synthetic_pect_raster(sY: int = 40, sX: int = 40, T: int = 500, num_defects: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic PECT raster scan (sY, sX, T) and Ground-Truth defect mask for testing.
    """
    raster = np.zeros((sY, sX, T), dtype=np.float32)
    mask = np.zeros((sY, sX), dtype=bool)

    # Time vector
    t = np.linspace(0, 1, T, dtype=np.float32)
    base_wave = np.sin(2 * np.pi * 5 * t) * np.exp(-3 * t)

    for y in range(sY):
        for x in range(sX):
            # Background noise
            noise = np.random.normal(0, 0.05, size=(T,)).astype(np.float32)
            raster[y, x, :] = base_wave + noise

    # Add artificial defects
    for d in range(num_defects):
        cy = np.random.randint(5, sY - 5)
        cx = np.random.randint(5, sX - 5)
        rad = np.random.randint(2, 4)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dy**2 + dx**2 <= rad**2:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < sY and 0 <= nx < sX:
                        mask[ny, nx] = True
                        defect_signal = 0.5 * np.exp(-5 * t) * np.cos(2 * np.pi * 12 * t)
                        raster[ny, nx, :] += defect_signal

    return raster, mask


def train_hespatial_dsbn(
    task_name: str = "Task_1",
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    seed: int = config.RANDOM_SEED,
):
    set_seed(seed)
    print(f"==================================================")
    print(f"🚀 Starting HeSpatial-DSBN Training for {task_name}")
    print(f"==================================================")

    task_info = config.TRANSFER_TASKS.get(task_name, config.TRANSFER_TASKS["Task_1"])
    src_sensor, src_mat = task_info["src_sensor"], task_info["src_mat"]
    tgt_sensor, tgt_mat = task_info["tgt_sensor"], task_info["tgt_mat"]

    print(f"🔹 Source Domain: {src_sensor.upper()} on {src_mat.upper()}")
    print(f"🔹 Target Domain: {tgt_sensor.upper()} on {tgt_mat.upper()}")

    # 1. Generate / Load Datasets
    print("\n📦 Loading PECT Rasters & Sampling Star Context...")
    sY, sX, T = 40, 40, config.TIME_SAMPLES
    raster_source, mask_source = generate_synthetic_pect_raster(sY, sX, T)
    raster_target, mask_target = generate_synthetic_pect_raster(sY, sX, T)

    sampler_s = StarContextSampler(raster_source, scales=config.SCALES)
    sampler_t = StarContextSampler(raster_target, scales=config.SCALES)

    all_centers = np.argwhere(np.ones((sY, sX), dtype=bool)).astype(np.int32)
    X_s, y_s = sampler_s.batch(all_centers)
    X_t, y_t = sampler_t.batch(all_centers)

    src_domain_id = config.DOMAIN_MAP.get((src_sensor, src_mat), 0)
    tgt_domain_id = config.DOMAIN_MAP.get((tgt_sensor, tgt_mat), 1)

    print(f"  Source Samples: {X_s.shape[0]} | Target Samples: {X_t.shape[0]}")
    print(f"  Context Input Tensor Shape: {X_s.shape} | Target Waveform Shape: {y_s.shape}")

    # 2. Build Model & Optimizer
    input_shape = (T, config.NUM_CONTEXT_CHANNELS)
    model, encoder, decoder, domain_head = build_hespatial_dsbn_model(
        input_shape=input_shape,
        num_domains=config.NUM_DOMAINS,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = HeSpatialDSBNLoss(
        lambda_mmd=config.LAMBDA_MMD,
        lambda_grl=config.LAMBDA_GRL,
        lambda_svdd=config.LAMBDA_SVDD,
    )

    # 3. Two-Stage Training Loop
    history = []
    best_target_cnr = -1.0
    svdd_center = None

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Ramp-up schedules for Stage 2
        if epoch <= config.WARMUP_EPOCHS:
            # Stage 1: Warm-up on Source
            alpha_t = 0.0
            k_ratio = config.K_START_RATIO
            grl_lambda = 0.0
            stage_name = "Stage 1 (Warmup)"
        else:
            # Stage 2: Adversarial & Ramp-up
            progress = (epoch - config.WARMUP_EPOCHS) / max(1, config.RAMP_EPOCHS)
            progress = min(1.0, progress)
            alpha_t = config.LAMBDA_REC_T * progress
            k_ratio = config.K_START_RATIO + (config.K_END_RATIO - config.K_START_RATIO) * progress
            grl_lambda = config.LAMBDA_GRL * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)
            stage_name = "Stage 2 (Adversarial Adaptation)"

        # Shuffle batches
        n_samples = len(X_s)
        indices_s = np.random.permutation(n_samples)
        indices_t = np.random.permutation(len(X_t))

        epoch_loss = 0.0
        epoch_rec_s = 0.0
        epoch_rec_t = 0.0
        epoch_mmd = 0.0
        epoch_adv = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx_s_b = indices_s[i : i + batch_size]
            idx_t_b = indices_t[i : i + batch_size]

            bx_s, by_s = X_s[idx_s_b], y_s[idx_s_b]
            bx_t, by_t = X_t[idx_t_b], y_t[idx_t_b]

            bd_s = np.full((len(bx_s),), src_domain_id, dtype=np.int32)
            bd_t = np.full((len(bx_t),), tgt_domain_id, dtype=np.int32)

            with tf.GradientTape() as tape:
                y_s_pred, y_t_pred, d_logits, z_s, z_t = model([bx_s, bx_t, bd_s, bd_t], training=True)

                if svdd_center is None:
                    svdd_center = compute_svdd_center(z_s)

                domain_labels = tf.concat([bd_s, bd_t], axis=0)

                losses_dict = loss_fn(
                    y_s_true=by_s,
                    y_s_pred=y_s_pred,
                    y_t_true=by_t,
                    y_t_pred=y_t_pred,
                    z_source=z_s,
                    z_target=z_t,
                    domain_logits=d_logits,
                    domain_labels=domain_labels,
                    alpha_t=alpha_t,
                    k_ratio=k_ratio,
                    svdd_center=svdd_center,
                )

                loss_value = losses_dict["total"]

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += float(loss_value)
            epoch_rec_s += float(losses_dict["rec_s"])
            epoch_rec_t += float(losses_dict["rec_t"])
            epoch_mmd += float(losses_dict["mmd"])
            epoch_adv += float(losses_dict["adv"])
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_rec_s = epoch_rec_s / max(1, n_batches)
        avg_rec_t = epoch_rec_t / max(1, n_batches)
        elapsed = time.time() - start_time

        # 4. Epoch Evaluation
        inf_model_t = build_inference_model(encoder, decoder, domain_id=tgt_domain_id)
        pred_y_t = inf_model_t.predict(X_t, verbose=0)
        metrics_t = evaluate_reconstruction_metrics(y_t.reshape(sY, sX, T), pred_y_t.reshape(sY, sX, T), mask_target)

        tgt_cnr = metrics_t.get("cnr", 0.0)
        tgt_auc = metrics_t.get("roc_auc", 0.0)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{stage_name}] ({elapsed:.1f}s) - "
            f"Loss: {avg_loss:.4f} | RecS: {avg_rec_s:.4f} | RecT: {avg_rec_t:.4f} | "
            f"Target CNR: {tgt_cnr:.3f} | Target ROC-AUC: {tgt_auc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "stage": stage_name,
            "loss": avg_loss,
            "rec_s": avg_rec_s,
            "rec_t": avg_rec_t,
            "target_cnr": tgt_cnr,
            "target_auc": tgt_auc,
        })

        # Save Best Model Checkpoint
        if tgt_cnr > best_target_cnr:
            best_target_cnr = tgt_cnr
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_hespatial_dsbn_{task_name}.h5")
            encoder.save_weights(ckpt_path.replace(".h5", "_encoder.h5"))
            decoder.save_weights(ckpt_path.replace(".h5", "_decoder.h5"))
            print(f"  ⭐ Saved Best Checkpoint! New High Target CNR: {best_target_cnr:.3f}")

    # 5. Save Final Training Logs
    log_file = os.path.join(config.LOG_DIR, f"training_log_{task_name}.json")
    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ HeSpatial-DSBN Training Complete! Best Target CNR: {best_target_cnr:.3f}")
    print(f"   Log File Saved: {log_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HeSpatial-DSBN for PECT Cross-Sensor Adaptation.")
    parser.add_argument("--task", type=str, default="Task_1", help="Task name (Task_1 to Task_6)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    train_hespatial_dsbn(
        task_name=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
