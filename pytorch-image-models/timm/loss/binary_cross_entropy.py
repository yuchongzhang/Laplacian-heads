"""Binary cross entropy with label smoothing. Copyright 2021 Ross Wightman."""
import torch
from torch import nn
from torch.nn import functional as F

class BinaryCrossEntropy(nn.Module):
    def __init__(self, smoothing=.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        if target.shape != x.shape:
            classes = x.shape[-1]
            off_value = self.smoothing / classes
            on_value = 1. - self.smoothing + off_value
            target = torch.full((x.shape[0], classes), off_value, device=x.device, dtype=x.dtype).scatter_(1, target.long().view(-1,1), on_value)
        return F.binary_cross_entropy_with_logits(x, target)
