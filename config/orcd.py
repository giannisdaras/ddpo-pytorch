import importlib.util
import os

import ml_collections


def _load_base():
    base_path = os.path.join(os.path.dirname(__file__), "base.py")
    spec = importlib.util.spec_from_file_location("base_config", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base config from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def compressibility():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # ORCD: 4x A100-SXM4-80GB via mit_preemptable.
    # Effective samples per epoch = 4 GPUs * 8 batch * 8 batches = 256  (parity with DGX 8-GPU run)
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 8

    # Effective train batch size = 4 GPUs * 4 batch * 4 accum = 64  (parity with DGX)
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 4

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config


def incompressibility():
    config = compressibility()
    config.reward_fn = "jpeg_incompressibility"
    return config


def aesthetic():
    config = compressibility()
    config.num_epochs = 200
    config.reward_fn = "aesthetic_score"
    config.train.gradient_accumulation_steps = 8  # harder reward, double accum
    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config


def clip_iqa():
    config = compressibility()
    config.reward_fn = "clip_iqa"
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 4
    return config


def brisque():
    config = compressibility()
    config.reward_fn = "brisque"
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 4
    return config


def get_config(name):
    return globals()[name]()
