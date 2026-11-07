# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
"""DeiT-B factories for CIFAR and ImageNet."""
import fnmatch
import re
from .vision_transformer import VisionTransformer

_MODELS = {}
for patch, size in [(4, 32), (16, 224)]:
    _MODELS[f'deit3_base_qk_norm_patch{patch}_{size}'] = (patch, size, 0)
    for heads in [1, 3, 6, 9, 11, 12]:
        _MODELS[f'deit3_base_{heads}L_patch{patch}_{size}'] = (patch, size, heads)

def list_models(filter='*'):
    return sorted(name for name in _MODELS if fnmatch.fnmatch(name, filter))

def is_model(name):
    return name in _MODELS

def safe_model_name(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def create_model(model_name, pretrained=False, checkpoint_path='', **kwargs):
    """Create a DeiT-B model; checkpoint_path loads a local state dictionary."""
    if pretrained:
        raise ValueError('Supply a local checkpoint_path instead of pretrained=True')
    if model_name not in _MODELS:
        raise ValueError(f'Unknown model {model_name!r}; choose from {list_models()}')
    patch, size, heads = _MODELS[model_name]
    defaults = dict(img_size=size, patch_size=patch, num_vanilla_heads=12-heads)
    defaults.update(kwargs)
    model = VisionTransformer(**defaults)
    if checkpoint_path:
        from ._helpers import load_checkpoint
        load_checkpoint(model, checkpoint_path)
    return model

from ._helpers import load_checkpoint, resume_checkpoint
