"""LayerNorm. Copyright 2022 Ross Wightman."""
from torch import nn
class LayerNorm(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True, **kwargs):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)
