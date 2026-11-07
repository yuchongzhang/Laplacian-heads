# Copyright 2020 Ross Wightman
"""Local checkpoint loading for model, optimizer, and AMP state."""
from collections import OrderedDict
import torch

def clean_state_dict(state):
    cleaned = OrderedDict()
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if key in cleaned:
            raise ValueError(f'Duplicate checkpoint key: {key}')
        cleaned[key] = value
    return cleaned

def load_checkpoint(model, checkpoint_path):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = saved.get('state_dict', saved.get('model', saved))
    model.load_state_dict(clean_state_dict(state), strict=True)
    return saved

def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    saved = load_checkpoint(model, checkpoint_path)
    if optimizer is not None and 'optimizer' in saved:
        optimizer.load_state_dict(saved['optimizer'])
    if loss_scaler is not None and loss_scaler.state_dict_key in saved:
        loss_scaler.load_state_dict(saved[loss_scaler.state_dict_key])
    return saved.get('epoch', -1) + 1
