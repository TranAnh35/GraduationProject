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
from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf

from . import config
from .dataloader import (
    StarContextSampler,
    get_normal_centers,
    generate_synthetic_pect_raster,
    load_dataset_raster,
)
from .losses import HeSpatialDSBNLoss, compute_svdd_center
from .model import build_hespatial_dsbn_model, build_inference_model
from .processing import evaluate_reconstruction_metrics


def set_seed(seed: int = 42):
    """Set random seeds for numpy and tensorflow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_hespatial_dsbn(
    task_name: str = "Task_1",
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    seed: int = config.RANDOM_SEED,
    use_real_data: bool = True,
    crack_size: str = "1mm",
):
    set_seed(seed)
    print(f"==================================================")
    print(f"🚀 Starting HeSpatial-DSBN Training for {task_name}")
    print(f"==================================================")

    # Resolve task info from available task tables
    task_tables = [config.TRANSFER_TASKS, config.CROSS_MATERIAL_TASKS, config.CROSS_SENSOR_TASKS]
    task_info = None
    for tbl in task_tables:
        if task_name in tbl:
            task_info = tbl[task_name]
            break
    if task_info is None:
        task_info = config.TRANSFER_TASKS.get("Task_1")

    src_sensor, src_mat = task_info["src_sensor"], task_info["src_mat"]
    tgt_sensor, tgt_mat = task_info["tgt_sensor"], task_info["tgt_mat"]

    print(f"🔹 Source Domain: {src_sensor.upper()} on {src_mat.upper()}")
    print(f"🔹 Target Domain: {tgt_sensor.upper()} on {tgt_mat.upper()}")
    print(f"🔹 Data Mode: {'Real TDMS Data (' + crack_size + ')' if use_real_data else 'Synthetic PECT Raster'}")

    # 1. Load Source and Target Datasets
    print("\n📦 Loading PECT Rasters & Sampling Star Context...")
    raster_source, mask_source = load_dataset_raster(
        sensor=src_sensor,
        material=src_mat,
        crack_size=crack_size,
        use_synthetic=not use_real_data,
        T=config.TIME_SAMPLES,
    )
    raster_target, mask_target = load_dataset_raster(
        sensor=tgt_sensor,
        material=tgt_mat,
        crack_size=crack_size,
        use_synthetic=not use_real_data,
        T=config.TIME_SAMPLES,
    )

    sY, sX, T = raster_source.shape
    tY, tX, _ = raster_target.shape

    sampler_s = StarContextSampler(raster_source, scales=config.SCALES)
    sampler_t = StarContextSampler(raster_target, scales=config.SCALES)

    # NORMAL-ONLY Source Training: Train reconstruction on defect-free normal pixels
    normal_centers_s = get_normal_centers(raster_source, material_name=src_mat, mask=mask_source)
    np.random.shuffle(normal_centers_s)

    # 80/20 Train-Val split on normal source samples for unsupervised model selection
    split_idx = max(1, int(len(normal_centers_s) * 0.8))
    train_centers_s = normal_centers_s[:split_idx]
    val_centers_s = normal_centers_s[split_idx:] if split_idx < len(normal_centers_s) else train_centers_s

    X_s_train, y_s_train = sampler_s.batch(train_centers_s)
    X_s_val, y_s_val = sampler_s.batch(val_centers_s)

    # Unlabeled Target Sampling: All spatial positions available for adaptation
    target_centers = np.argwhere(np.ones((tY, tX), dtype=bool)).astype(np.int32)
    X_t, y_t = sampler_t.batch(target_centers)

    src_domain_id = config.DOMAIN_MAP.get((src_sensor.lower(), src_mat.lower()), 0)
    tgt_domain_id = config.DOMAIN_MAP.get((tgt_sensor.lower(), tgt_mat.lower()), 1)

    print(f"  Source Normal Samples (Train / Val): {X_s_train.shape[0]} / {X_s_val.shape[0]}")
    print(f"  Target Unlabeled Samples: {X_t.shape[0]}")
    print(f"  Context Tensor Shape: {X_s_train.shape} | Target Waveform Shape: {y_s_train.shape}")

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
    best_val_loss = float("inf")
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
            # Ganin & Lempitsky 2016 DANN schedule: lambda_p = 2 / (1 + exp(-10 * p)) - 1
            grl_lambda = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
            stage_name = "Stage 2 (Adversarial Adaptation)"

        # Update GRL layer reversal factor dynamically
        try:
            grl_layer = model.get_layer("grl")
            if hasattr(grl_layer, "set_lambda"):
                grl_layer.set_lambda(grl_lambda)
        except Exception:
            pass

        # Shuffle batches
        n_samples = len(X_s_train)
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
            idx_t_b = indices_t[i % len(X_t) : (i % len(X_t)) + len(idx_s_b)]
            if len(idx_t_b) < len(idx_s_b):
                idx_t_b = np.random.choice(len(X_t), size=len(idx_s_b), replace=False)

            bx_s, by_s = X_s_train[idx_s_b], y_s_train[idx_s_b]
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
                    current_lambda_grl=config.LAMBDA_GRL * grl_lambda,
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

        # 4. Unsupervised Validation for Model Selection (No Target Defect Labels)
        inf_model_s = build_inference_model(encoder, decoder, domain_id=src_domain_id)
        pred_y_s_val = inf_model_s.predict(X_s_val, verbose=0)
        val_loss_s = float(np.mean(np.abs(y_s_val - pred_y_s_val)))

        # Target Monitoring (Logged for inspection only, not used for model selection)
        inf_model_t = build_inference_model(encoder, decoder, domain_id=tgt_domain_id)
        pred_y_t = inf_model_t.predict(X_t, verbose=0)
        metrics_t = evaluate_reconstruction_metrics(y_t.reshape(tY, tX, T), pred_y_t.reshape(tY, tX, T), mask_target)

        tgt_cnr = metrics_t.get("cnr", 0.0)
        tgt_auc = metrics_t.get("roc_auc", 0.0)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{stage_name}] ({elapsed:.1f}s) - "
            f"Loss: {avg_loss:.4f} | RecS: {avg_rec_s:.4f} | ValS: {val_loss_s:.4f} | "
            f"GRL_λ: {grl_lambda:.3f} | Target CNR: {tgt_cnr:.3f} | Target ROC-AUC: {tgt_auc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "stage": stage_name,
            "loss": avg_loss,
            "rec_s": avg_rec_s,
            "val_loss_s": val_loss_s,
            "rec_t": avg_rec_t,
            "grl_lambda": grl_lambda,
            "target_cnr": tgt_cnr,
            "target_auc": tgt_auc,
        })

        # Save Best Model Checkpoint based on Source Validation Loss (Unsupervised Model Selection)
        if val_loss_s < best_val_loss:
            best_val_loss = val_loss_s
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_hespatial_dsbn_{task_name}.h5")
            encoder.save_weights(ckpt_path.replace(".h5", "_encoder.h5"))
            decoder.save_weights(ckpt_path.replace(".h5", "_decoder.h5"))
            print(f"  ⭐ Saved Best Checkpoint! New Low Source Val Loss: {best_val_loss:.4f}")

    # 5. Save Final Training Logs
    log_file = os.path.join(config.LOG_DIR, f"training_log_{task_name}.json")
    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ HeSpatial-DSBN Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"   Log File Saved: {log_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HeSpatial-DSBN for PECT Cross-Sensor Adaptation.")
    parser.add_argument("--task", type=str, default="Task_1", help="Task name (e.g. Task_1 to Task_6, CM_1..6, CS_AL_1..6)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--real_data", action="store_true", default=True, help="Use real TDMS data")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    parser.add_argument("--crack", type=str, default="1mm", help="Crack size for real TDMS (1mm to 5mm)")
    args = parser.parse_args()

    use_real = not args.synthetic

    train_hespatial_dsbn(
        task_name=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        use_real_data=use_real,
        crack_size=args.crack,
    )
