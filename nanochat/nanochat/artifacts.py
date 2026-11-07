"""Directories for language checkpoints and evaluation outputs."""

import os
from nanochat.common import get_base_dir


def get_checkpoint_dir(model_tag):
    return os.path.join(get_base_dir(), "base_checkpoints", model_tag)


def get_base_eval_dir(model_tag):
    return os.path.join(get_base_dir(), "base_eval", model_tag)
