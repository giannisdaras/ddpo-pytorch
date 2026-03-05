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


def _base_h100():
    """LS6 gpu-h100: 1 node, 2x H100 80GB — exact ORCD 4-GPU effective scale.

    Effective samples/epoch: 2 * 8 * 16 = 256  (parity with ORCD)
    Effective train batch:   2 * 4 * 8  = 64   (parity with ORCD)
    """
    config = base.get_config()
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 16

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 8

    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    return config


def _base_a100():
    """LS6 gpu-a100: 1 node, 3x A100 40GB — approximate ORCD effective scale.

    Effective samples/epoch: 3 * 8 * 8 = 192  (75% of ORCD's 256)
    Effective train batch:   3 * 4 * 4 = 48   (75% of ORCD's 64)
    Note: 3 GPUs can't evenly divide 256 or 64; using ORCD-equivalent per-device
    params (8 batches, grad_acc=4) so the train assertion passes.
    Constraint: samples_per_device (64) % (train_batch * grad_acc) (16) == 0 ✓
    """
    config = base.get_config()
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 8

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 4

    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    return config


def compressibility_h100():
    config = _base_h100()
    config.reward_fn = "jpeg_compressibility"
    return config


def compressibility_a100():
    config = _base_a100()
    config.reward_fn = "jpeg_compressibility"
    return config


def get_config(name):
    return globals()[name]()
