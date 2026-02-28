#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 06:00:00
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

# Config — override with DDPO_CONFIG / DDPO_RUN_NAME env vars
CONFIG=${DDPO_CONFIG:-clip_iqa_2gpu}
RUN_NAME=${DDPO_RUN_NAME:-orcd_${CONFIG}}

# Checkpoint dir = logdir / run_name  (matches train.py ProjectConfiguration)
LOGDIR=$REPO/logs/$RUN_NAME
CKPT_DIR=$LOGDIR/$RUN_NAME

echo "=== DDPO Launch (mit_normal_gpu / 2× H200) ==="
echo "Config:   $CONFIG"
echo "Run name: $RUN_NAME"
echo "Node:     $(hostname)"
echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -2)"
echo "Date:     $(date)"

mkdir -p $REPO/logs

# Login to wandb and HF
conda run -n ddpo wandb login $WANDB_API_KEY --relogin
conda run -n ddpo huggingface-cli login --token $HF_TOKEN

cd $REPO

# Auto-resume from latest checkpoint if the run already has one
RESUME_FLAG=""
if [ -d "$CKPT_DIR" ] && ls "$CKPT_DIR"/checkpoint_* 2>/dev/null | head -1 > /dev/null; then
    RESUME_FLAG="--config.resume_from $CKPT_DIR"
    echo "Resuming from checkpoint in $CKPT_DIR"
else
    echo "Starting fresh run"
fi

accelerate launch --num_processes 2 --mixed_precision fp16 \
    scripts/train.py \
    --config config/orcd.py:$CONFIG \
    --config.run_name $RUN_NAME \
    --config.logdir $LOGDIR \
    $RESUME_FLAG
