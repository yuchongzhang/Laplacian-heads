# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import lru_cache
from typing import Callable, Optional

import torch
from torchvision.datasets.vision import StandardTransform


logger = logging.getLogger("dinov3")


def get_target_transform(dataset) -> Optional[Callable]:
    if hasattr(dataset, "transforms"):
        if isinstance(dataset.transforms, StandardTransform):
            return dataset.transforms.target_transform
        raise ValueError("Dataset has a non-standard .transforms property")
    if hasattr(dataset, "target_transform"):
        return dataset.target_transform
    return None


@lru_cache(maxsize=1)
def get_labels(dataset) -> torch.Tensor:
    """
    Get the labels of a classification dataset, as a Tensor, using the `get_targets` method
    if it is present or loading the labels one by one with `get_target`, if it exists.
    If the dataset has a target transform, iterate over the whole dataset to get the
    transformed labels for each element, then stack them as a torch tensor.
    """
    logger.info("Getting dataset labels ...")
    if hasattr(dataset, "get_targets") or hasattr(dataset, "get_target"):
        if hasattr(dataset, "get_targets"):  # Returns a np.array
            labels = dataset.get_targets()
        elif hasattr(dataset, "get_target"):
            labels = [dataset.get_target(i) for i in range(len(dataset))]
        target_transform = get_target_transform(dataset)
        if target_transform is not None:
            labels = [target_transform(label) for label in labels]
    else:
        # Target transform is applied in this case
        labels = [dataset[i][1] for i in range(len(dataset))]
    return torch.stack([torch.tensor(label, dtype=int) for label in labels])


def get_num_classes(dataset) -> int:
    """
    Get the labels of a dataset and compute the number of classes
    """
    labels = get_labels(dataset)
    if len(labels.shape) > 1:
        return int(labels.shape[1])
    return int(labels.max() + 1)
