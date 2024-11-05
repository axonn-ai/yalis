import torch
from litgpt.model import Config
from pathlib import Path
import sys
from litgpt.utils import load_checkpoint
from external.model import GPT


def get_model(model_id, fabric, litgpt_checkpoint_directory, random_init=False):
    with fabric.init_module(empty_init=True):
        checkpoint_dir = Path(litgpt_checkpoint_directory)
        config = Config.from_file(checkpoint_dir / "model_config.yaml")
        model = GPT(config, gradient_checkpointing=False)
    model = fabric.setup_module(model)
    if not random_init:
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        load_checkpoint(fabric, model, checkpoint_path)
    model.eval()
    return model
