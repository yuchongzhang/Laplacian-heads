# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import torch
import torch.nn.functional as F
from torch import nn


def reduce_loss(loss, reduction) -> torch.Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = nn._reduction.get_enum(reduction)
    #  None: 0, element-wise mean: 1, sum: 2
    assert reduction_enum in [0, 1, 2]
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None) -> torch.Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        class_weight=None,
        loss_weight=1.0,
        reduction="mean",
        avg_factor=None,
        ignore_index=255,
        avg_non_ignore=False,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore

    def forward(self, pred, label):
        loss = F.cross_entropy(pred, label, weight=self.class_weight, reduction="none", ignore_index=self.ignore_index)

        if (self.avg_factor is None) and self.avg_non_ignore and self.reduction == "mean":
            avg_factor = label.numel() - (label == self.ignore_index).sum().item()
        else:
            avg_factor = None

        loss = weight_reduce_loss(loss, weight=self.weight, reduction=self.reduction, avg_factor=avg_factor)
        return self.loss_weight * loss
