# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import numpy as np

logger = logging.getLogger("dinov3")




def linear_warmup_cosine_decay(
    start: float,
    peak: float,
    end: float,
    warmup_iterations: int,
    total_iterations: int,
    cosine_iterations: int | None = None,
) -> np.ndarray:
    """
    Create a learning rate schedule with linear warmup, a cosine, and an optional constant part in the end.

    Args:
        start (float): Initial learning rate.
        peak (float): Learning rate after linear warmup.
        end (float): Final learning rate after cosine.
        warmup_iterations (int): Number of iterations for linear warmup.
        total_iterations (int): Total number of iterations for the schedule.
        cosine_iterations (int | None): Number of iterations for cosine.
            If None, cosine part will be over remaining iterations after warmup.
    Returns:
        np.ndarray: Learning rate schedule as a numpy array.
    """
    linear = np.linspace(start, peak, warmup_iterations, endpoint=False)
    if cosine_iterations is None:
        cosine_iterations = total_iterations - warmup_iterations
    cosine = np.cos(np.linspace(0, np.pi, cosine_iterations))
    cosine = (cosine + 1) / 2
    cosine = (peak - end) * cosine + end
    remaining_iterations = total_iterations - cosine_iterations - warmup_iterations
    assert remaining_iterations >= 0
    constant = np.full((remaining_iterations,), fill_value=end)
    return np.concatenate([linear, cosine, constant])
