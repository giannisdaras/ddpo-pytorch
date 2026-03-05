#!/bin/bash
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH -A ASC25048
#SBATCH --output=/scratch/07362/gdaras/ddpo-pytorch/logs/ddpo_%j.out
#SBATCH --job-name=ddpo

set -e

SCRATCH=/scratch/07362/gdaras
REPO=$SCRATCH/ddpo-pytorch
CONDA=$HOME/miniconda3/envs/ddpo/bin

source $REPO/secrets/env.sh

# Config — override with DDPO_CONFIG env var
CONFIG=${DDPO_CONFIG:-saturation}
RUN_NAME=${DDPO_RUN_NAME:-vista_${CONFIG}}

echo "=== DDPO Vista Launch ==="
echo "Config:   $CONFIG"
echo "Run name: $RUN_NAME"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Date:     $(date)"

mkdir -p $REPO/logs

$CONDA/wandb login $WANDB_API_KEY --relogin
$CONDA/huggingface-cli login --token $HF_TOKEN

cd $REPO

$CONDA/accelerate launch --num_processes 1 --mixed_precision fp16 \
    scripts/train.py \
    --config config/vista.py:$CONFIG \
    --config.run_name $RUN_NAME \
    --config.logdir $REPO/logs/$RUN_NAME
