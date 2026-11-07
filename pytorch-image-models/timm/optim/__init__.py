"""AdamW and LAMB with DeiT weight-decay parameter groups."""
from torch import nn
from torch.optim import AdamW
from .lamb import Lamb
from ._param_groups import param_groups_weight_decay

def create_optimizer_v2(model, opt='adamw', lr=1e-3, weight_decay=0., eps=None, betas=None):
    if opt not in ('adamw', 'lamb'):
        raise ValueError('opt must be adamw or lamb')
    if isinstance(model, nn.Module):
        skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else ()
        params = param_groups_weight_decay(model, weight_decay, skip) if weight_decay else model.parameters()
    else:
        params = model
    kwargs = dict(lr=lr, weight_decay=0.)
    if eps is not None:
        kwargs['eps'] = eps
    if betas is not None:
        kwargs['betas'] = tuple(betas)
    return (AdamW if opt == 'adamw' else Lamb)(params, **kwargs)
