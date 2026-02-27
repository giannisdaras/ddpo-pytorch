from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob_tpu import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob_tpu import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image

# Torchrun TPU: bind each rank to one chip before torch_xla initializes.
if os.environ.get("PJRT_DEVICE", "").upper() == "TPU" and "LOCAL_RANK" in os.environ:
    local_rank = os.environ["LOCAL_RANK"]
    os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
    os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1")
    os.environ["TPU_VISIBLE_CHIPS"] = local_rank

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None
try:
    from peft import LoraConfig
except ImportError:
    LoraConfig = None

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # TPU training path using Accelerate+xla backend.
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")
    is_tpu = accelerator.device.type == "xla"
    # TPU collectives via Accelerate can become a bottleneck in this script's
    # host-reward path. Default to CPU-side reward aggregation on TPU; allow
    # opting back into Accelerate gather via env if needed.
    tpu_skip_gather = is_tpu and os.environ.get("DDPO_TPU_USE_ACCELERATE_GATHER", "0") != "1"

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if not is_tpu:
        if accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

    # Move unet/text encoder to device. On TPU, keep VAE on CPU and decode latents
    # on host for reward computation to avoid TPU->host image sync stalls.
    if is_tpu:
        pipeline.vae.to("cpu", dtype=torch.float32)
        pipeline.vae.eval()
        tpu_reward_vae = pipeline.vae
    else:
        pipeline.vae.to(accelerator.device, dtype=inference_dtype)
        tpu_reward_vae = None
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    lora_backend = None
    if config.use_lora:
        # Set correct lora layers
        if hasattr(pipeline.unet, "add_adapter"):
            if LoraConfig is None:
                raise ImportError(
                    "TPU LoRA path requires `peft` with modern diffusers. Install peft (e.g. `pip install peft`)."
                )
            pipeline.unet.requires_grad_(False)
            pipeline.unet.add_adapter(
                LoraConfig(
                    r=4,
                    lora_alpha=4,
                    init_lora_weights="gaussian",
                    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                )
            )
            unet = pipeline.unet
            lora_backend = "peft"
        else:
            lora_attn_procs = {}
            for name in pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = pipeline.unet.config.block_out_channels[block_id]

                try:
                    lora_attn_procs[name] = LoRAAttnProcessor(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                    )
                except TypeError:
                    lora_attn_procs[name] = LoRAAttnProcessor()
            pipeline.unet.set_attn_processor(lora_attn_procs)

            # Legacy path for older diffusers where attn processors are modules.
            class _Wrapper(AttnProcsLayers):
                def forward(self, *args, **kwargs):
                    return pipeline.unet(*args, **kwargs)

            unet = _Wrapper(pipeline.unet.attn_processors)
            lora_backend = "legacy_attn_procs"
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora:
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora:
            if lora_backend == "legacy_attn_procs":
                tmp_unet = UNet2DConditionModel.from_pretrained(
                    config.pretrained.model,
                    revision=config.pretrained.revision,
                    subfolder="unet",
                )
                tmp_unet.load_attn_procs(input_dir)
                models[0].load_state_dict(
                    AttnProcsLayers(tmp_unet.attn_processors).state_dict()
                )
                del tmp_unet
            else:
                pipeline.unet.load_attn_procs(input_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        if is_tpu:
            raise ValueError("8-bit Adam is not supported on TPU. Set config.train.use_8bit_adam=False.")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found for optimizer.")

    optimizer = optimizer_cls(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Keep reward execution asynchronous on GPU/CPU. TPU follows a separate
    # delayed-reward path below to avoid per-batch host-sync stalls.
    use_async_rewards = not is_tpu
    executor = futures.ThreadPoolExecutor(max_workers=2) if use_async_rewards else None
    # TPU fallback: inline per-batch rewards (debug mode only).
    tpu_inline_rewards = os.environ.get("DDPO_TPU_INLINE_REWARDS", "0") == "1"
    # Debug-only knob; disabled by default because explicit barriers can deadlock
    # on some TPU runtime versions.
    tpu_sync_every_batch = os.environ.get("DDPO_TPU_SYNC_EVERY_BATCH", "0") == "1"
    # Random per-sample timestep shuffling creates many unique timestep patterns
    # that can trigger excessive XLA recompiles. Keep deterministic timestep order
    # on TPU unless explicitly re-enabled.
    tpu_shuffle_timesteps = os.environ.get("DDPO_TPU_SHUFFLE_TIMESTEPS", "0") == "1"

    def tpu_to_cpu(tensor):
        return tensor.detach().cpu()

    def prepare_reward_images(tensor_batch):
        if not is_tpu:
            return tensor_batch
        # `tensor_batch` is latent output on TPU path.
        latents_cpu = tpu_to_cpu(tensor_batch)
        with torch.no_grad():
            decoded = tpu_reward_vae.decode(
                (latents_cpu / tpu_reward_vae.config.scaling_factor).to(
                    dtype=torch.float32
                ),
                return_dict=False,
            )[0]
        return (decoded / 2 + 0.5).clamp(0, 1)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] sampling start")
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []
        tpu_image_batches = []
        tpu_prompt_batches = []
        tpu_metadata_batches = []
        tpu_batch_size = config.sample.batch_size
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[
                    prompt_fn(**config.prompt_fn_kwargs)
                    for _ in range(config.sample.batch_size)
                ]
            )

            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            # sample
            with autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="latent" if is_tpu else "pt",
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            if use_async_rewards:
                # compute rewards asynchronously
                rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
                # yield to to make sure reward computation starts
                time.sleep(0)
            else:
                if tpu_inline_rewards:
                    host_images = prepare_reward_images(images)
                    rewards, _ = reward_fn(host_images, prompts, prompt_metadata)
                    reward_device = "cpu" if tpu_skip_gather else accelerator.device
                    rewards = torch.as_tensor(rewards, device=reward_device)
                else:
                    # TPU fast path: delay reward eval until all batches are sampled.
                    # Materialize small latent outputs to CPU now to avoid reward-stage
                    # XLA host-sync stalls.
                    tpu_image_batches.append(tpu_to_cpu(images))
                    tpu_prompt_batches.append(prompts)
                    tpu_metadata_batches.append(prompt_metadata)
                    rewards = None
                    if is_tpu and xm is not None and tpu_sync_every_batch:
                        xm.mark_step()

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed (non-TPU async path)
        if use_async_rewards:
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                rewards, reward_metadata = sample["rewards"].result()
                # accelerator.print(reward_metadata)
                sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
        else:
            # TPU path: evaluate rewards per sampled batch with explicit stage logs.
            if not tpu_inline_rewards:
                if accelerator.is_local_main_process:
                    logger.info(f"[epoch {epoch}] reward eval start ({len(samples)} batches)")
                for i, (sample, image_batch, prompt_batch, metadata_batch) in enumerate(
                    zip(samples, tpu_image_batches, tpu_prompt_batches, tpu_metadata_batches)
                ):
                    t0 = time.time()
                    host_images = prepare_reward_images(image_batch)
                    t1 = time.time()
                    rewards, _ = reward_fn(host_images, prompt_batch, metadata_batch)
                    t2 = time.time()
                    reward_device = "cpu" if tpu_skip_gather else accelerator.device
                    sample["rewards"] = torch.as_tensor(rewards, device=reward_device)
                    if accelerator.is_local_main_process:
                        logger.info(
                            f"[epoch {epoch}] reward batch {i+1}/{len(samples)} "
                            f"copy_s={t1 - t0:.2f} reward_s={t2 - t1:.2f}"
                        )
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] rewards ready")

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] samples collated")

        # On TPU, image logging can stall due device->host sync; keep scalar logging only.
        if not is_tpu:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, image in enumerate(images):
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                accelerator.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{i}.jpg"),
                                caption=f"{prompt:.25} | {reward:.2f}",
                            )
                            for i, (prompt, reward) in enumerate(
                                zip(prompts, rewards)
                            )  # only log rewards from process 0
                        ],
                    },
                    step=global_step,
                )
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] pre-gather")

        if is_tpu and xm is not None:
            # Flush outstanding lazy XLA work before gather/host reads; otherwise
            # the first collective can block for a very long compile+materialize
            # phase after sampling.
            xm.mark_step()

        # gather rewards across processes
        if tpu_skip_gather:
            rewards = samples["rewards"].detach().cpu().numpy()
        elif accelerator.num_processes == 1:
            rewards = samples["rewards"].detach().cpu().numpy()
        else:
            rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] rewards gathered")

        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            if tpu_skip_gather:
                prompt_ids = samples["prompt_ids"].detach().cpu().numpy()
            elif accelerator.num_processes == 1:
                prompt_ids = samples["prompt_ids"].detach().cpu().numpy()
            else:
                prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] advantages computed")

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] training start")
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.from_numpy(np.random.permutation(total_batch_size)).to(
                device=accelerator.device, dtype=torch.long
            )
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample.
            # Disabled on TPU by default to avoid pathological recompilation.
            if (not is_tpu) or tpu_shuffle_timesteps:
                perms_np = np.argsort(
                    np.random.uniform(size=(total_batch_size, num_timesteps)), axis=1
                ).astype(np.int64)
                perms = torch.from_numpy(perms_np).to(accelerator.device)
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None],
                        perms,
                    ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + config.sample.guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        if is_tpu and xm is not None:
                            xm.mark_step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
        if accelerator.is_local_main_process:
            logger.info(f"[epoch {epoch}] training done")

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    if xm is None:
        raise ImportError(
            "scripts/train_tpu.py requires torch_xla. Install torch_xla[tpu] on the TPU VM."
        )
    app.run(main)
