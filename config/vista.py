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


def _base_1gpu():
    """1-node 1-GPU (GraceHopper H100) config preserving ORCD 4-GPU effective scale.

    Effective samples/epoch: 1 * 8 * 32 = 256  (parity with ORCD)
    Effective train batch:   1 * 4 * 16 = 64   (parity with ORCD)
    """
    config = base.get_config()
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 32

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 16

    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    return config


def saturation():
    config = _base_1gpu()
    config.reward_fn = "saturation"
    return config


def entropy():
    config = _base_1gpu()
    config.reward_fn = "entropy"
    return config


def get_config(name):
    return globals()[name]()
