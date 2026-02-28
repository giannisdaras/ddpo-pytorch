#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --output=/home/%u/orcd/scratch/ddpo-pytorch/logs/ddpo_%j.out
#SBATCH --job-name=ddpo

set -e

SCRATCH=$HOME/orcd/scratch
REPO=$SCRATCH/ddpo-pytorch

# Load env
source $REPO/secrets/env.sh
module load miniforge/23.11.0-0
module load cuda/12.4.0
conda activate ddpo

# Config — override with DDPO_CONFIG env var, default compressibility
CONFIG=${DDPO_CONFIG:-compressibility}
RUN_NAME=${DDPO_RUN_NAME:-orcd_${CONFIG}}

echo "=== DDPO Launch ==="
echo "Config:   $CONFIG"
echo "Run name: $RUN_NAME"
echo "Node:     $(hostname)"
echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4)"
echo "Date:     $(date)"

mkdir -p $REPO/logs

# Login to wandb and HF
conda run -n ddpo wandb login $WANDB_API_KEY --relogin
conda run -n ddpo huggingface-cli login --token $HF_TOKEN

cd $REPO

accelerate launch --num_processes 4 --mixed_precision fp16 \
    scripts/train.py \
    --config config/orcd.py:$CONFIG \
    --config.run_name $RUN_NAME \
    --config.logdir $REPO/logs/$RUN_NAME
