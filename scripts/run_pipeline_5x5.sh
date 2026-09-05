#!/bin/bash
# ==============================================================================
# Unified 5x5 PECT-JEPA: Automated Train & Evaluate Pipeline
# Usage:
#     bash scripts/run_pipeline_5x5.sh [EXP_NAME] [EPOCHS] [BATCH_SIZE] [DATA_DIR]
# Example:
#     bash scripts/run_pipeline_5x5.sh pect_jepa_5x5_v1 50 256 data
# ==============================================================================

set -e

EXP_NAME=${1:-"pect_jepa_5x5_v1"}
EPOCHS=${2:-50}
BATCH_SIZE=${3:-256}
DATA_DIR=${4:-"data"}

echo "======================================================================"
echo "  Starting 5x5 PECT-JEPA Fresh Training: ${EXP_NAME}"
echo "  Epochs: ${EPOCHS} | Batch Size: ${BATCH_SIZE} | Data: ${DATA_DIR}"
echo "======================================================================"

# Step 1: Self-Supervised Pretraining
python -m src.PECT_JEPA.spatiotemporal_5x5.train \
    --data_dir "${DATA_DIR}" \
    --exp_name "${EXP_NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --split_protocol compound_ood \
    --resample_mode linear \
    --in_channels 128 \
    --crop_border 10 \
    --normalization global_peak \
    --learning_rate 3e-4 \
    --loss_type smooth_l1 \
    --cov_weight 0.5 \
    --var_weight 1.0 \
    --mixed_precision True \
    --use_tensorboard True \
    --device cuda

echo ""
echo "======================================================================"
echo "  Training Completed! Launching Downstream Evaluation & Probing..."
echo "======================================================================"

CHECKPOINT_PATH="experiments/5x5/${EXP_NAME}/checkpoints/best_model_5x5.pt"
SPLIT_SUMMARY_PATH="experiments/5x5/${EXP_NAME}/${EXP_NAME}_split_summary.json"
OUTPUT_EVAL_DIR="evaluation_results/5x5/${EXP_NAME}"

# Step 2: Downstream Evaluation (OOD Defect Contrast CNR + Lift-Off CKA)
python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split_summary "${SPLIT_SUMMARY_PATH}" \
    --output_dir "${OUTPUT_EVAL_DIR}" \
    --data_dir "${DATA_DIR}" \
    --crop_border 10 \
    --eval_liftoff \
    --device cuda

echo ""
echo "======================================================================"
echo "  Pipeline Finished Successfully!"
echo "  - Checkpoints: experiments/5x5/${EXP_NAME}/checkpoints/"
echo "  - Evaluation:  ${OUTPUT_EVAL_DIR}/"
echo "  - Report:      ${OUTPUT_EVAL_DIR}/evaluation_report.json"
echo "======================================================================"
