#!/usr/bin/env bash
set -euo pipefail

# Example launcher for the pathology JPEG recompression model.
# Update DATASET_ROOT to your local dataset path before running.

DATASET_ROOT="${DATASET_ROOT:-/path/to/pathology_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-./compress_output}"
QUALITY="${QUALITY:-75}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NET_SIZE="${NET_SIZE:-B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fractral_tmi_q${QUALITY}}"

python -u train_lossless_jpeg_trans.py \
  --cuda \
  --dataset "${DATASET_ROOT}" \
  --quality "${QUALITY}" \
  --batch-size "${BATCH_SIZE}" \
  --test-batch-size "${TEST_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --net "${NET_SIZE}" \
  --method pathology_jpeg_trans \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}"
