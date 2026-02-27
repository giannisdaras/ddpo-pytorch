#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/launch_tpu.sh <config_path_or_name> [extra args...]"
  echo "Example: scripts/launch_tpu.sh config/dgx.py:compressibility --config.num_epochs=1"
  exit 1
fi

CONFIG="$1"
shift

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export PYTHONUNBUFFERED=1
# Keep disabled by default; DDPO internals use integer timestep indexing
# that can fail under global BF16 rewrite.
export XLA_USE_BF16="${XLA_USE_BF16:-0}"
# Sync every N denoising steps to avoid giant deferred XLA graphs at reward
# handoff while preserving throughput. Tuned for v6e-8.
export DDPO_TPU_SYNC_EVERY_STEP="${DDPO_TPU_SYNC_EVERY_STEP:-2}"

# If these are not set, libtpu discovers all local chips.
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-}"
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS:-}"

NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch_xla.distributed.xla_spawn') else 1)"; then
  "$PYTHON_BIN" -m torch_xla.distributed.xla_spawn --num_processes "$NPROC" scripts/train_tpu.py --config "$CONFIG" "$@"
else
  if [[ "$NPROC" -gt 1 ]]; then
    if command -v torchrun >/dev/null 2>&1; then
      torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_tpu.py --config "$CONFIG" "$@"
      exit 0
    fi
    if "$PYTHON_BIN" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch.distributed.run') else 1)"; then
      "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_tpu.py --config "$CONFIG" "$@"
      exit 0
    fi
    echo "torch_xla.distributed.xla_spawn is unavailable, and torchrun was not found."
    echo "Set NPROC=1 or install torchrun for multi-process TPU launch."
    exit 1
  fi
  "$PYTHON_BIN" -u scripts/train_tpu.py --config "$CONFIG" "$@"
fi
