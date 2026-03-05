#!/bin/bash
# Launch DDPO on torralba-3090-4 (4x RTX 3090, GPUs 1-4)
set -e

SCRATCH=/data/vision/torralba/scratch/gdaras/scratch_nfs
REPO=$SCRATCH/rl-predictions/ddpo-pytorch
LOGDIR=$SCRATCH/rl-predictions/ddpo-runs

export CUDA_VISIBLE_DEVICES=1,2,3,4
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/wandb_cache

source $REPO/secrets/env.sh

mkdir -p $SCRATCH/wandb $SCRATCH/wandb_cache

CONFIG=${DDPO_CONFIG:-pickscore}
RUN_NAME=${DDPO_RUN_NAME:-torralba_${CONFIG}_$(date +%Y.%m.%d_%H.%M.%S)}

echo "=== DDPO Torralba Launch ==="
echo "Config:   $CONFIG"
echo "Run name: $RUN_NAME"
echo "Node:     $(hostname)"
echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4)"
echo "Date:     $(date)"

mkdir -p $LOGDIR/logs

cd $REPO

$SCRATCH/u/envs/hidden_vocab/bin/accelerate launch \
    --num_processes 4 --mixed_precision fp16 \
    scripts/train.py \
    --config config/torralba.py:$CONFIG \
    --config.run_name $RUN_NAME \
    --config.logdir $LOGDIR \
    2>&1 | tee $LOGDIR/logs/${CONFIG}_$(date +%Y%m%d_%H%M%S).log
