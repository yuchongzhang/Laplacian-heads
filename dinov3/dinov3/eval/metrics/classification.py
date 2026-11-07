# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy


def build_classification_metric(*, num_classes):
    """ImageNet micro-averaged top-1 and top-5 accuracy."""
    ks = (1, 5) if num_classes >= 5 else (1,)
    return MetricCollection({
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average="micro")
        for k in ks
    })
