"""Image transformations. Copyright Ross Wightman."""
import math
import random
import warnings
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

def str_to_interp_mode(name):
    return InterpolationMode(name)

def interp_mode_to_str(mode):
    return mode.value

_RANDOM_INTERPOLATION = (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3. / 4., 4. / 3.),
            interpolation='bilinear',
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        img_w, img_h = F.get_image_size(img)
        area = img_w * img_h

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if target_w <= img_w and target_h <= img_h:
                i = random.randint(0, img_h - target_h)
                j = random.randint(0, img_w - target_w)
                return i, j, target_h, target_w

        # Fallback to central crop
        in_ratio = img_w / img_h
        if in_ratio < min(ratio):
            target_w = img_w
            target_h = int(round(target_w / min(ratio)))
        elif in_ratio > max(ratio):
            target_h = img_h
            target_w = int(round(target_h * max(ratio)))
        else:  # whole image
            target_w = img_w
            target_h = img_h
        i = (img_h - target_h) // 2
        j = (img_w - target_w) // 2
        return i, j, target_h, target_w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([interp_mode_to_str(x) for x in self.interpolation])
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
