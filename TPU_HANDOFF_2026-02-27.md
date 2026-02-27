# TPU DDPO Handoff (2026-02-27)

## Scope
This note captures TPU debugging + run state for DDPO compressibility parity runs on the active TPU VM.

## Current Environment
- Project: `ambient-488017`
- TPU VM: `trc-spot-v6e-ew4b`
- Zone: `europe-west4-a`
- Accelerator: `v6e-8`
- Runtime observed: Python 3.12, `torch 2.9.0`, `torch_xla 2.9.0`

## Active Run (do not kill)
- Run name: `tpu_compressibility_parity_fix11h_20260227_193254`
- Log: `/home/giannis/projects/rl-predictions/ddpo-pytorch/logs_tpu_runs/tpu_compressibility_parity_fix11h_20260227_193254.log`
- Launch pattern:
  - `NPROC=1`
  - `DDPO_TPU_SYNC_EVERY_STEP=2`
  - `DDPO_TPU_SYNC_EVERY_BATCH=1`
  - `--config.mixed_precision=no`
  - parity overrides: `sample.batch_size=4`, `sample.num_batches_per_epoch=8`, `train.batch_size=2`, `train.gradient_accumulation_steps=4`, `num_epochs=100`, `save_freq=1`
- Last observed status (2026-02-27 19:41 UTC): still in `[epoch 0] sampling start`.
- Updated status (2026-02-27 19:43 UTC): passed reward + gather and entered `[epoch 0] training start`.

## DDPO Parity Constraints (kept)
These were preserved intentionally (no effective parameter drift):
- Total samples per epoch: `256`
- Total train batch size: `64`
- Inner epochs: `1`
- Sample steps: `50`

## Key Findings
1. **Sampling host-sync bottleneck was real and severe** before patching.
   - In focused benchmark, `images.detach().cpu()` took ~311s per batch.
2. **Adding periodic XLA sync inside denoising fixed the huge copy stall**.
   - `DDPO_TPU_SYNC_EVERY_STEP=2` gave best tradeoff in local test:
     - `sync_every=2`: ~`6.0s` total sample+copy (batch=4)
     - `sync_every=1`: ~`22.8s`
     - `sync_every=10`: ~`7.6s`
3. **`pre-gather` hang after reward stage** was addressed by TPU-path changes:
   - explicit `xm.mark_step()` before gather
   - TPU-side gather bypass for rewards/prompt IDs by default (`DDPO_TPU_USE_ACCELERATE_GATHER!=1`)
   - In run `fix11g`, this progressed through:
     - `[epoch 0] pre-gather`
     - `[epoch 0] rewards gathered`
     - `[epoch 0] advantages computed`
     - `[epoch 0] training start`
4. **Current dominant bottleneck moved to early training step compile**.
   - In `fix11g`, first training timestep logged at `~6m51s` for step 1/50.
5. **Hypothesis:** dynamic timestep shuffling causes TPU recompilation pressure.
   - TPU default changed to disable per-sample timestep permutation unless `DDPO_TPU_SHUFFLE_TIMESTEPS=1`.

## Important Runs Snapshot
- `fix11g` (`tpu_compressibility_parity_fix11g_20260227_190706`):
  - Reached `rewards gathered` and `training start`.
  - First training step still very slow.
- `fix11h` (`tpu_compressibility_parity_fix11h_20260227_193254`):
  - Running now with timestep-shuffle disabled by default.
  - Confirmed progression through:
    - `[epoch 0] reward eval start`
    - `[epoch 0] pre-gather`
    - `[epoch 0] rewards gathered`
    - `[epoch 0] training start`

## Code Changes Made (local repo)
- `scripts/train_tpu.py`
  - TPU reward path and delayed reward evaluation from latents.
  - TPU sync controls (`DDPO_TPU_SYNC_EVERY_BATCH`, pre-gather `mark_step`).
  - TPU gather bypass defaults for rewards/prompt IDs.
  - TPU default: no timestep permutation unless `DDPO_TPU_SHUFFLE_TIMESTEPS=1`.
- `ddpo_pytorch/diffusers_patch/pipeline_with_logprob_tpu.py`
  - Added per-step TPU sync control; default `DDPO_TPU_SYNC_EVERY_STEP=2`.
- `ddpo_pytorch/diffusers_patch/ddim_with_logprob_tpu.py`
  - TPU-safe DDIM + logprob scheduler patch.
- `scripts/launch_tpu.sh`
  - TPU launcher with `xla_spawn` fallback chain and default sync env.
- `config/dgx.py`
  - Replaced removed `imp` usage with `importlib.util` loader for Python 3.12.
- `README.md`
  - TPU usage section.

## Monitoring Commands
```bash
# Process + key progress markers
gcloud compute tpus tpu-vm ssh trc-spot-v6e-ew4b --project=ambient-488017 --zone=europe-west4-a --worker=0 --command='\
  date; \
  ps -eo pid,etimes,pcpu,pmem,cmd | grep -E "train_tpu.py|launch_tpu.sh" | grep -v grep; \
  grep -n "\\[epoch 0\\]\\|\\[epoch 1\\]\\|rewards gathered\\|training start" \
    /home/giannis/projects/rl-predictions/ddpo-pytorch/logs_tpu_runs/tpu_compressibility_parity_fix11h_20260227_193254.log | tail -n 120'
```

## Next Steps for Continuation
1. Let `fix11h` finish epoch 0 and confirm whether training-step latency improved vs `fix11g`.
2. If training step-1 remains pathological, test reducing training-loop XLA barriers (`xm.mark_step`) frequency in `train_tpu.py` while preserving correctness.
3. Keep effective DDPO parity values unchanged unless explicitly instructed otherwise.
4. If a relaunch is needed, keep the same hyperparameter parity overrides and use the same TPU VM/zone.

## Cautions
- TPU VM SSH is flaky; retries are common.
- Avoid starting extra TPU Python processes while a run is active (can lock `/dev/vfio/*`).
- `NPROC=8` via `torch.distributed.run` currently duplicates independent runs (accelerate process accounting mismatch), so prefer `NPROC=1` for this code path.
