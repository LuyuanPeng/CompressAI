#!/bin/bash
set -e

# Output directory for evaluation results
RESULTS_DIR="results/eval_finetuned/NVSPrior/tcoms_0719"
mkdir -p "$RESULTS_DIR"

# Dataset for evaluation
# EVAL_DATASET="/home/luyuan/Data/tcoms2024/0719/slc_camera"
EVAL_DATASET="/home/luyuan/Data/image_compression/NVSPrior/tcoms19_test"

# Evaluate mbt2018-mean 
python3 -m compressai.utils.eval_model checkpoint "$EVAL_DATASET" \
  -a mbt2018-mean \
  -p mbt2018-mean_mse_q2_lambda0.1800.pth.tar \
  --cuda \
  --output_directory "$RESULTS_DIR" \
  --output-file mbt2018-mean_mse180

# # Evaluate mbt2018-mean (lambda=0.1)
# python3 -m compressai.utils.eval_model checkpoint "$EVAL_DATASET" \
#   -a mbt2018-mean \
#   -p mbt2018-mean_tcoms_lambda0.1.pth.tar \
#   --cuda \
#   --output_directory "$RESULTS_DIR" \
#   --output-file mbt2018-mean_lambda_1e-1
# python3 -m compressai.utils.eval_model checkpoint "$EVAL_DATASET" \
#   -a mbt2018-mean \
#   -p mbt2018-mean_tcoms_lambda1.pth.tar \
#   --cuda \
#   --output_directory "$RESULTS_DIR" \
#   --output-file mbt2018-mean_lambda_1e0

# Evaluate mbt2018 (lambda=0.001)
# python3 -m compressai.utils.eval_model checkpoint "$EVAL_DATASET" \
#   -a mbt2018 \
#   -p mbt2018_tcoms_lambda0.001.pth.tar \
#   --cuda \
#   --output_directory "$RESULTS_DIR" \
#   --output-file mbt2018_lambda_1e-3
