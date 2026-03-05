#!/bin/bash
#SBATCH -p allnodes
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 3-00:00:00
#SBATCH --output=/datastor1/gdaras/ddpo-pytorch/logs/ddpo_%j.out
#SBATCH --job-name=ddpo

set -e

REPO=/datastor1/gdaras/ddpo-pytorch
CONDA=/datastor1/gdaras/miniforge3/envs/ddpo/bin

source $REPO/secrets/env.sh

CONFIG=${DDPO_CONFIG:-clip_score}
RUN_NAME=${DDPO_RUN_NAME:-utcs_${CONFIG}}

echo "=== DDPO UTCS Launch ==="
echo "Config:   $CONFIG"
echo "Run name: $RUN_NAME"
echo "Node:     $(hostname)"
echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4)"
echo "Date:     $(date)"

mkdir -p $REPO/logs

$CONDA/wandb login $WANDB_API_KEY --relogin
$CONDA/huggingface-cli login --token $HF_TOKEN

cd $REPO

$CONDA/accelerate launch --num_processes 4 --mixed_precision fp16 \
    scripts/train.py \
    --config config/utcs.py:$CONFIG \
    --config.run_name $RUN_NAME \
    --config.logdir $REPO/logs/$RUN_NAME
