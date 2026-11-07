# Copyright 2019 Ross Wightman
"""Crops and augmentations for fixed-size ImageNet images."""
import math
from torchvision import transforms
from .auto_augment import auto_augment_transform
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .transforms import RandomResizedCropAndInterpolation, str_to_interp_mode

def create_transform(input_size=(3,224,224), is_training=False, use_prefetcher=False,
                     auto_augment='3a', color_jitter=.3, interpolation='bicubic',
                     mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, crop_pct=.875,
                     scale=(.08,1.), ratio=(.75,4/3), hflip=.5):
    size = input_size[-1] if isinstance(input_size, (tuple,list)) else input_size
    if is_training:
        operations = [RandomResizedCropAndInterpolation(size, scale=scale, ratio=ratio, interpolation=interpolation)]
        if hflip:
            operations.append(transforms.RandomHorizontalFlip(hflip))
        if auto_augment:
            operations.append(auto_augment_transform(auto_augment, dict(
                translate_const=int(size*.45), img_mean=tuple(min(255, round(255*x)) for x in mean))))
        if color_jitter:
            operations.append(transforms.ColorJitter(*((float(color_jitter),)*3)))
    else:
        operations = [transforms.Resize(math.floor(size/(crop_pct or .875)), interpolation=str_to_interp_mode(interpolation)),
                      transforms.CenterCrop(size)]
    operations += [transforms.PILToTensor()] if use_prefetcher else [transforms.ToTensor(), transforms.Normalize(mean,std)]
    return transforms.Compose(operations)
