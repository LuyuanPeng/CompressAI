#!/bin/bash
set -e

# Dataset paths (update these to change datasets easily)
DATASET_PATH="/home/luyuan/Data/image_compression/coral_camera"
TESTSET_PATH="/home/luyuan/Data/image_compression/coral_camera/test"

# Output directory for evaluation results
RESULTS_DIR="results/eval_finetuned/coral"
mkdir -p "$RESULTS_DIR"

# Remove old checkpoints
rm -f checkpoint.pth.tar checkpoint_best_loss.pth.tar mbt2018_best_*.pth.tar cheng2020-anchor_best_*.pth.tar mbt2018-mean_best_*.pth.tar

# Standard lambda table for CompressAI (MSE only)
# MSE_LAMBDAS=(0.0018 0.0035 0.0067 0.0130 0.0250 0.0483 0.0932 0.1800)
MSE_LAMBDAS=(0.1800)

# Parse command-line arguments
MODE="both" # Default mode is both finetune and eval
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --finetune-only)
      MODE="finetune"
      shift
      ;;
    --eval-only)
      MODE="eval"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set dataset tag for naming
DATASET_TAG="Coral"


# MSE metric loop (quality 1-8)
for i in {0..1}; do
  lmbda=${MSE_LAMBDAS[$i]}
  quality=$((i+1))
  lmbda_name=$(printf "%.4f" "$lmbda")
  # Only use MSE metric

  if [[ "$MODE" == "both" || "$MODE" == "finetune" ]]; then
    # mbt2018-mean
    python3 examples/train.py \
      --model mbt2018-mean \
      --dataset "$DATASET_PATH" \
      --epochs 100 \
      --batch-size 8 \
      --cuda \
      --save \
      --lambda "$lmbda" \
      --patch-size 128 128
    mv checkpoint_best_loss.pth.tar mbt2018-mean_${DATASET_TAG}_mse_q${quality}_lambda${lmbda_name}.pth.tar
    rm -f checkpoint.pth.tar

    # # cheng2020-anchor (only for quality 1-6)
    # if [[ $quality -le 6 ]]; then
    #   python3 examples/train.py \
    #     --model cheng2020-anchor \
    #     --dataset "$DATASET_PATH" \
    #     --epochs 100 \
    #     --batch-size 8 \
    #     --cuda \
    #     --save \
    #     --lambda "$lmbda" \
    #     --patch-size 128 128
    #   mv checkpoint_best_loss.pth.tar cheng2020-anchor_mse_q${quality}_lambda${lmbda_name}.pth.tar
    #   rm -f checkpoint.pth.tar
    # fi
  fi

  if [[ "$MODE" == "both" || "$MODE" == "eval" ]]; then
    # mbt2018-mean
    python3 -m compressai.utils.eval_model checkpoint "$TESTSET_PATH" \
      -a mbt2018-mean \
      -p mbt2018-mean_${DATASET_TAG}_mse_q${quality}_lambda${lmbda_name}.pth.tar \
      --cuda \
      --output_directory "$RESULTS_DIR" \
      --output-file mbt2018-mean_mse_q${quality}.json

    # # cheng2020-anchor (only for quality 1-6)
    # if [[ $quality -le 6 ]]; then
    #   python3 -m compressai.utils.eval_model checkpoint "$TESTSET_PATH" \
    #     -a cheng2020-anchor \
    #     -p cheng2020-anchor_mse_q${quality}_lambda${lmbda_name}.pth.tar \
    #     --cuda \
    #     --output_directory "$RESULTS_DIR" \
    #     --output-file cheng2020-anchor_mse_q${quality}.json
    # fi
  fi
done

# ## MS-SSIM metric loop removed
# for i in {0..7}; do
#   lmbda=${MS_SSIM_LAMBDAS[$i]}
#   quality=$((i+1))
#   lmbda_name=$(printf "%.2f" "$lmbda")
#   metric="ms-ssim"

#   if [[ "$MODE" == "both" || "$MODE" == "finetune" ]]; then
#     # mbt2018-mean
#     python3 examples/train.py \
#       --model mbt2018-mean \
#       --dataset "$DATASET_PATH" \
#       --epochs 100 \
#       --batch-size 8 \
#       --cuda \
#       --save \
#       --lambda "$lmbda" \
#       --metric "$metric" \
#       --patch-size 128 128
#     mv checkpoint_best_loss.pth.tar mbt2018-mean_${metric}_q${quality}_lambda${lmbda_name}.pth.tar
#     rm -f checkpoint.pth.tar

#     # cheng2020-anchor (only for quality 1-6)
#     if [[ $quality -le 6 ]]; then
#       python3 examples/train.py \
#         --model cheng2020-anchor \
#         --dataset "$DATASET_PATH" \
#         --epochs 100 \
#         --batch-size 8 \
#         --cuda \
#         --save \
#         --lambda "$lmbda" \
#         --metric "$metric" \
#         --patch-size 128 128
#       mv checkpoint_best_loss.pth.tar cheng2020-anchor_${metric}_q${quality}_lambda${lmbda_name}.pth.tar
#       rm -f checkpoint.pth.tar
#     fi
#   fi

#   if [[ "$MODE" == "both" || "$MODE" == "eval" ]]; then
#     # mbt2018-mean
#     python3 -m compressai.utils.eval_model checkpoint "$TESTSET_PATH" \
#       -a mbt2018-mean \
#       -p mbt2018-mean_${metric}_q${quality}_lambda${lmbda_name}.pth.tar \
#       --cuda \
#       --output_directory "$RESULTS_DIR" \
#       --output-file mbt2018-mean_${metric}_q${quality}.json

#     # cheng2020-anchor (only for quality 1-6)
#     if [[ $quality -le 6 ]]; then
#       python3 -m compressai.utils.eval_model checkpoint "$TESTSET_PATH" \
#         -a cheng2020-anchor \
#         -p cheng2020-anchor_${metric}_q${quality}_lambda${lmbda_name}.pth.tar \
#         --cuda \
#         --output_directory "$RESULTS_DIR" \
#         --output-file cheng2020-anchor_${metric}_q${quality}.json
#     fi
#   fi
# done
