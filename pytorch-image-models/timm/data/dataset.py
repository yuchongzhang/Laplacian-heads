# Copyright 2019 Ross Wightman
"""Class-folder datasets with natural filename ordering."""
from pathlib import Path
from torchvision.datasets import ImageFolder
from timm.utils.misc import natural_key

class ImageDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        old_classes = self.classes
        self.classes = sorted(old_classes, key=natural_key)
        self.class_to_idx = {label:i for i,label in enumerate(self.classes)}
        self.samples = sorted([(name,self.class_to_idx[old_classes[label]]) for name,label in self.samples],
                              key=lambda sample:natural_key(sample[0]))
        self.imgs = self.samples
        self.targets = [label for _,label in self.samples]

def create_dataset(name='', root='', split='train'):
    if name:
        raise ValueError('Only class-folder image datasets are supported')
    return ImageDataset(Path(root)/split)
