"""RandAugment and 3-Augment image transforms.

RandAugment operations adapted from:
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
3-Augment based on:
https://github.com/facebookresearch/deit/blob/main/README_revenge.md

Copyright 2019, Ross Wightman
"""
import math
import random
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

_LEVEL_DENOM = 10.  # denominator for conversion from 'Mx' magnitude scale to fractional aug level for op arguments

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

if hasattr(Image, "Resampling"):
    _RANDOM_INTERPOLATION = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.Resampling.BICUBIC
else:
    _RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.BICUBIC

def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', _DEFAULT_INTERPOLATION)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    return interpolation

def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)

def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    if _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    return img.rotate(degrees, resample=kwargs['resample'])

def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)

def invert(img, **__):
    return ImageOps.invert(img)

def equalize(img, **__):
    return ImageOps.equalize(img)

def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)

def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)

    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)

    return img

def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)

def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)

def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)

def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)

def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)

def gaussian_blur_rand(img, factor, **__):
    radius_min = 0.1
    radius_max = 2.0
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(radius_min, radius_max * factor)))
    return img

def desaturate(img, factor, **_):
    factor = min(1., max(0., 1. - factor))
    # enhance factor 0 = grayscale, 1.0 = no-change
    return ImageEnhance.Color(img).enhance(factor)

def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v

def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _LEVEL_DENOM) * 30.
    level = _randomly_negate(level)
    return level,

def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _LEVEL_DENOM) * 1.8 + 0.1,

def _minmax_level_to_arg(level, _hparams, min_val=0., max_val=1.0, clamp=True):
    level = (level / _LEVEL_DENOM)
    level = min_val + (max_val - min_val) * level
    if clamp:
        level = max(min_val, min(max_val, level))
    return level,

def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return level,

def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _LEVEL_DENOM) * translate_pct
    level = _randomly_negate(level)
    return level,

def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4),

def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return min(256, int((level / _LEVEL_DENOM) * 256)),

def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return min(128, int((level / _LEVEL_DENOM) * 110)),

LEVEL_TO_ARG = {'AutoContrast': None, 'Equalize': None, 'Invert': None, 'Rotate': _rotate_level_to_arg, 'Posterize': _posterize_level_to_arg, 'Solarize': _solarize_level_to_arg, 'SolarizeAdd': _solarize_add_level_to_arg, 'Color': _enhance_level_to_arg, 'Contrast': _enhance_level_to_arg, 'Brightness': _enhance_level_to_arg, 'Sharpness': _enhance_level_to_arg, 'ShearX': _shear_level_to_arg, 'ShearY': _shear_level_to_arg, 'TranslateXRel': _translate_rel_level_to_arg, 'TranslateYRel': _translate_rel_level_to_arg, 'Desaturate': partial(_minmax_level_to_arg, min_val=0.5, max_val=1.0), 'GaussianBlurRand': _minmax_level_to_arg}

NAME_TO_OP = {'AutoContrast': auto_contrast, 'Equalize': equalize, 'Invert': invert, 'Rotate': rotate, 'Posterize': posterize, 'Solarize': solarize, 'SolarizeAdd': solarize_add, 'Color': color, 'Contrast': contrast, 'Brightness': brightness, 'Sharpness': sharpness, 'ShearX': shear_x, 'ShearY': shear_y, 'TranslateXRel': translate_x_rel, 'TranslateYRel': translate_y_rel, 'Desaturate': desaturate, 'GaussianBlurRand': gaussian_blur_rand}

class AugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)
        self.magnitude_max = self.hparams.get('magnitude_max', None)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            # magnitude randomization enabled
            if self.magnitude_std == float('inf'):
                # inf == uniform sampling
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
        # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        fs = self.__class__.__name__ + f'(name={self.name}, p={self.prob}'
        fs += f', m={self.magnitude}, mstd={self.magnitude_std}'
        if self.magnitude_max is not None:
            fs += f', mmax={self.magnitude_max}'
        fs += ')'
        return fs

_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    # 'Cutout'  # NOTE I've implement this as random erasing separately
]

def auto_augment_policy_3a(hparams):
    policy = [
        [('Solarize', 1.0, 5)],  # 128 solarize threshold @ 5 magnitude
        [('Desaturate', 1.0, 10)],  # grayscale at 10 magnitude
        [('GaussianBlurRand', 1.0, 10)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc

class AutoAugment:

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img

    def __repr__(self):
        fs = self.__class__.__name__ + '(policy='
        for p in self.policy:
            fs += '\n\t['
            fs += ', '.join([str(op) for op in p])
            fs += ']'
        fs += ')'
        return fs

def rand_augment_ops(
        magnitude: Union[int, float] = 10,
        prob: float = 0.5,
        hparams: Optional[Dict] = None,
        transforms: Optional[Union[Dict, List]] = None,
):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(
        name, prob=prob, magnitude=magnitude, hparams=hparams) for name in transforms]

class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        for op in ops:
            img = op(img)
        return img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs

def auto_augment_transform(config_str, hparams=None):
    if config_str != '3a':
        raise ValueError('Only the 3a ImageNet policy is supported')
    return AutoAugment(auto_augment_policy_3a(hparams or _HPARAMS_DEFAULT))

def rand_augment_transform(config_str, hparams=None):
    if config_str != 'rand-m9-mstd0.5':
        raise ValueError('Only rand-m9-mstd0.5 is supported')
    hparams = dict(hparams or _HPARAMS_DEFAULT)
    hparams.setdefault('magnitude_std', .5)
    return RandAugment(rand_augment_ops(magnitude=9, prob=.5, hparams=hparams), 2)
