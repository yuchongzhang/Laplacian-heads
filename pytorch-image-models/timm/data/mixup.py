"""Batch Mixup and CutMix. Copyright 2019 Ross Wightman."""
import numpy as np
import torch

def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)

def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh

def cutmix_bbox_and_lam(img_shape, lam):
    yl, yu, xl, xu = rand_bbox(img_shape, lam)
    area = (yu-yl)*(xu-xl)
    return (yl,yu,xl,xu), 1. - area / float(img_shape[-2]*img_shape[-1])


class Mixup:
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., prob=1., switch_prob=.5,
                 label_smoothing=.1, num_classes=1000):
        self.mixup_alpha, self.cutmix_alpha = mixup_alpha, cutmix_alpha
        self.mix_prob, self.switch_prob = prob, switch_prob
        self.label_smoothing, self.num_classes = label_smoothing, num_classes
        self.mixup_enabled = True

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0. should be true."
            lam = float(lam_mix)
        return lam, use_cutmix
    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam
    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Mixup requires an even batch size'
        lam = self._mix_batch(x)
        return x, mixup_target(target, self.num_classes, lam, self.label_smoothing)


class FastCollateMixup(Mixup):
    def _mix_batch_collate(self, output, batch):
        batch_size = len(batch)
        lam, use_cutmix = self._params_per_batch()
        is_np = isinstance(batch[0][0], np.ndarray)

        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                output.shape,
                lam,
            )
        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy() if is_np else mixed.clone()  # don't want to modify the original while iterating
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                else:
                    if is_np:
                        mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                        np.rint(mixed, out=mixed)
                    else:
                        mixed = mixed.float() * lam + batch[j][0].float() * (1 - lam)
                        torch.round(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8)) if is_np else mixed.byte()
        return lam

    def __call__(self, batch):
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Mixup requires an even batch size'
        output = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        lam = self._mix_batch_collate(output, batch)
        target = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        return output, mixup_target(target, self.num_classes, lam, self.label_smoothing)
