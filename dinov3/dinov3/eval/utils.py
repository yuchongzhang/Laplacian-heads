# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import gc
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torchmetrics import Metric

from dinov3.eval.accumulators import NoOpAccumulator, ResultsAccumulator
from dinov3.logging import MetricLogger

logger = logging.getLogger("dinov3")


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n, autocast_ctx, reshape=False, return_class_token=True, norm=True):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n = n  # Layer indices (Sequence) or n last layers (int) to take
        self.autocast_ctx = autocast_ctx
        self.reshape = reshape
        self.return_class_token = return_class_token
        self.norm = norm

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images,
                    n=self.n,
                    reshape=self.reshape,
                    return_class_token=self.return_class_token,
                    norm=self.norm,
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, Metric],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    accumulate_results: bool = False,
):
    gc.collect()  # Avoids garbage collection errors in DataLoader workers
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    accumulator_class = ResultsAccumulator if accumulate_results else NoOpAccumulator
    accumulators = {k: accumulator_class() for k in postprocessors.keys()}

    # Dataset needs to be wrapped in fairvit.data.adapters.DatasetWithEnumeratedTargets
    for samples, (index, targets), *_ in metric_logger.log_every(data_loader, 10, header):
        samples, targets, index = samples[index >= 0], targets[index >= 0], index[index >= 0]
        if len(index) == 0:
            continue

        outputs = model(samples.to(device))
        index = index.to(device)
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)
            accumulators[k].update(preds=metric_inputs["preds"], target=metric_inputs["target"], index=index)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # accumulator.accumulate() returns None for the NoOpAccumulator
    accumulated_results = {k: accumulator.accumulate() for k, accumulator in accumulators.items()}

    return metric_logger_stats, stats, accumulated_results


def save_results(
    preds: torch.Tensor,
    target: torch.Tensor,
    output_dir: str,
    filename_suffix: Optional[str] = None,
) -> None:
    """
    Helper to save predictions from a model and their associated targets, aligned by their index
    """
    filename_suffix = "" if filename_suffix is None else f"_{filename_suffix}"
    preds_filename = f"preds{filename_suffix}.npy"
    target_filename = f"target{filename_suffix}.npy"
    preds_path = os.path.join(output_dir, preds_filename)
    target_path = os.path.join(output_dir, target_filename)
    logger.info(f"Saving to {preds_path}")
    np.save(preds_path, preds.cpu().numpy())
    logger.info(f"Saving to {target_path}")
    np.save(target_path, target.cpu().numpy())
