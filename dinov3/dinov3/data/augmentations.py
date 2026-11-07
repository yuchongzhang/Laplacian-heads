# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import torch
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")


        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        self.global_transfo1 = v2.Compose(
            [color_jittering, global_transfo1_extra, self.normalize]
        )
        self.global_transfo2 = v2.Compose(
            [color_jittering, global_transfo2_extra, self.normalize]
        )
        self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}


        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]


        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops

        return output
