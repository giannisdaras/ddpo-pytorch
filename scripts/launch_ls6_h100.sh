#!/bin/bash
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH -A MLL
#SBATCH --output=/scratch/07362/gdaras/ddpo-pytorch/logs/ddpo_%j.out
#SBATCH --job-name=ddpo

set -e

SCRATCH=/scratch/07362/gdaras
REPO=/home1/07362/gdaras/ddpo-pytorch
CONDA=/work/07362/gdaras/ls6/miniconda3/envs/ddpo/bin

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/wandb_cache

source $REPO/secrets/env.sh

CONFIG=${DDPO_CONFIG:-compressibility_h100}
RUN_NAME=${DDPO_RUN_NAME:-ls6_h100_${CONFIG}_$(date +%Y.%m.%d_%H.%M.%S)}

echo "=== DDPO LS6 H100 Launch ==="
echo "Config:   $CONFIG"
echo "Run name: $RUN_NAME"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date:     $(date)"

mkdir -p $SCRATCH/ddpo-pytorch/logs $SCRATCH/hf_cache $SCRATCH/wandb $SCRATCH/wandb_cache

$CONDA/wandb login $WANDB_API_KEY --relogin
$CONDA/huggingface-cli login --token $HF_TOKEN

cd $REPO

$CONDA/accelerate launch --num_processes 2 --mixed_precision fp16 \
    scripts/train.py \
    --config config/ls6.py:$CONFIG \
    --config.run_name $RUN_NAME \
    --config.logdir $SCRATCH/ddpo-pytorch/logs/$RUN_NAME
