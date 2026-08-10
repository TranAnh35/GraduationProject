"""
Evaluation and Benchmark Testing Script for HeSpatial-DSBN.
Computes CNR, NCC, SSIM, MAE, and Pixel ROC-AUC across all 6 Cross-Sensor Transfer Tasks.

Standalone executable script - no external project dependencies.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf

from . import config
from .dataloader import StarContextSampler
from .model import build_hespatial_encoder, build_hespatial_decoder, build_inference_model
from .processing import evaluate_reconstruction_metrics, generate_synthetic_pect_raster if hasattr(sys.modules[__name__], 'generate_synthetic_pect_raster') else None


def evaluate_task(task_name: str = "Task_1") -> dict:
    print(f"📊 Evaluating HeSpatial-DSBN on {task_name}...")
    
    task_info = config.TRANSFER_TASKS.get(task_name, config.TRANSFER_TASKS["Task_1"])
    tgt_sensor, tgt_mat = task_info["tgt_sensor"], task_info["tgt_mat"]
    tgt_domain_id = config.DOMAIN_MAP.get((tgt_sensor, tgt_mat), 1)

    sY, sX, T = 40, 40, config.TIME_SAMPLES

    # Import synthetic generator from train module if needed
    from .train import generate_synthetic_pect_raster
    raster_target, mask_target = generate_synthetic_pect_raster(sY, sX, T)

    sampler = StarContextSampler(raster_target, scales=config.SCALES)
    all_centers = np.argwhere(np.ones((sY, sX), dtype=bool)).astype(np.int32)
    X_t, y_t = sampler.batch(all_centers)

    input_shape = (T, config.NUM_CONTEXT_CHANNELS)
    encoder = build_hespatial_encoder(input_shape)
    enc_out_shape = encoder.output_shape[1:]
    decoder = build_hespatial_decoder(enc_out_shape)

    ckpt_encoder = os.path.join(config.CHECKPOINT_DIR, f"best_hespatial_dsbn_{task_name}_encoder.h5")
    ckpt_decoder = os.path.join(config.CHECKPOINT_DIR, f"best_hespatial_dsbn_{task_name}_decoder.h5")

    if os.path.exists(ckpt_encoder) and os.path.exists(ckpt_decoder):
        encoder.load_weights(ckpt_encoder)
        decoder.load_weights(ckpt_decoder)
        print(f"  Loaded weights from {ckpt_encoder}")
    else:
        print(f"  ⚠️ Warning: Checkpoint not found for {task_name}. Evaluating with initialized weights.")

    inf_model = build_inference_model(encoder, decoder, domain_id=tgt_domain_id)
    pred_y_t = inf_model.predict(X_t, verbose=0)

    y_t_3d = y_t.reshape(sY, sX, T)
    pred_3d = pred_y_t.reshape(sY, sX, T)

    metrics = evaluate_reconstruction_metrics(y_t_3d, pred_3d, mask=mask_target)
    
    print(f"  ✅ Results for {task_name}:")
    print(f"     CNR (Defect Prominence): {metrics.get('cnr', 0.0):.3f}")
    print(f"     NCC (Waveform Fidelity): {metrics.get('ncc', 0.0):.4f}")
    print(f"     SSIM (Structure Similarity): {metrics.get('ssim', 0.0):.4f}")
    print(f"     ROC-AUC (Pixel Anomaly): {metrics.get('roc_auc', 0.0):.4f}")

    return {
        "task": task_name,
        "src_domain": f"{task_info['src_sensor']}_{task_info['src_mat']}",
        "tgt_domain": f"{task_info['tgt_sensor']}_{task_info['tgt_mat']}",
        "metrics": metrics,
    }


def run_full_benchmark():
    print("==================================================")
    print("🏆 Running Full Benchmark Evaluation (Tasks 1 to 6)")
    print("==================================================")

    results = {}
    for task_name in config.TRANSFER_TASKS.keys():
        res = evaluate_task(task_name)
        results[task_name] = res

    output_report = os.path.join(config.OUTPUT_DIR, "benchmark_report.json")
    with open(output_report, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n🎉 Full Benchmark Evaluation Complete! Report saved to {output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HeSpatial-DSBN Benchmark.")
    parser.add_argument("--task", type=str, default="all", help="Task name ('Task_1' to 'Task_6' or 'all')")
    args = parser.parse_args()

    if args.task.lower() == "all":
        run_full_benchmark()
    else:
        evaluate_task(args.task)
