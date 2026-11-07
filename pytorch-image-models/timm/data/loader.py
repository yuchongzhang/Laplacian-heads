"""Image loading, uint8 mixup collation, and CUDA prefetch.

Adapted from timm. Copyright 2019 Ross Wightman.
"""
from contextlib import suppress
from functools import partial
import numpy as np
import torch
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .distributed_sampler import RepeatAugSampler, OrderedDistributedSampler
from .transforms_factory import create_transform

def fast_collate(batch):
    targets = torch.tensor([item[1] for item in batch], dtype=torch.int64)
    images = torch.zeros((len(batch), *batch[0][0].shape), dtype=torch.uint8)
    for i, (image, _) in enumerate(batch):
        images[i] += torch.from_numpy(image) if isinstance(image, np.ndarray) else image
    return images, targets

class PrefetchLoader:
    def __init__(self, loader, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, device=torch.device('cuda')):
        self.loader, self.device = loader, device
        self.mean = torch.tensor([x*255 for x in mean], device=device, dtype=torch.float32).view(1,3,1,1)
        self.std = torch.tensor([x*255 for x in std], device=device, dtype=torch.float32).view(1,3,1,1)

    def __iter__(self):
        stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        context = partial(torch.cuda.stream, stream=stream) if stream else suppress
        previous = None
        for images, labels in self.loader:
            with context():
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                images = images.float().sub_(self.mean).div_(self.std)
            if previous is not None:
                yield previous
            if stream:
                torch.cuda.current_stream(device=self.device).wait_stream(stream)
            previous = images, labels
        if previous is not None:
            yield previous

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    np.random.seed(info.seed % (2**32-1))

def create_loader(dataset, input_size, batch_size, is_training=False, num_workers=4,
                  distributed=False, num_aug_repeats=0, collate_fn=None, device=torch.device('cuda'),
                  mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, **transform_args):
    dataset.transform = create_transform(input_size, is_training=is_training, use_prefetcher=True,
                                         mean=mean, std=std, **transform_args)
    sampler = None
    if distributed:
        if is_training:
            sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats) if num_aug_repeats else torch.utils.data.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)
    elif num_aug_repeats:
        sampler = RepeatAugSampler(dataset, num_replicas=1, rank=0, num_repeats=num_aug_repeats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=is_training and sampler is None, sampler=sampler, num_workers=num_workers,
        collate_fn=collate_fn or fast_collate, pin_memory=device.type=='cuda', drop_last=is_training,
        persistent_workers=num_workers>0, worker_init_fn=_worker_init)
    if not len(loader):
        raise ValueError('Training dataset is too small for this batch size or repeated augmentation; reduce --batch-size or set --aug-repeats 0')
    return PrefetchLoader(loader, mean, std, device)
