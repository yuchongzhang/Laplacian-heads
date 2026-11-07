"""Checkpoint, dataset, and feature-hook helpers shared by the six measurements."""
import json
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from timm import create_model

DATASETS = {
    'cifar10': (10, 32, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': (100, 32, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'imagenet': (1000, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def load_model(name, checkpoint, dataset, device):
    """Strictly load CIFAR/timm, compiled, or DDP state dictionaries."""
    model = create_model(name, pretrained=False, num_classes=DATASETS[dataset][0])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = saved.get('state_dict', saved.get('model', saved))
    cleaned = {}
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if key in cleaned:
            raise ValueError(f'Duplicate checkpoint key after prefix removal: {key}')
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return model.eval().to(device)


def dataset_for(name, root, metric):
    """Construct the metric-specific evaluation transforms."""
    _, size, mean, std = DATASETS[name]
    if metric in ('pca', 'nc'):
        ops = [transforms.Resize((size, size))]
        mean, std = (0.5,) * 3, (0.5,) * 3
        preprocessing = f'square_resize_{size}; normalize(mean=.5,std=.5)'
    elif name == 'imagenet':
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        preprocessing = 'resize_short_side_256; center_crop_224; imagenet_normalize'
    else:
        ops = []
        preprocessing = f'{name}_normalize; native_32x32'
    transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if name == 'imagenet':
        dataset = datasets.ImageFolder(Path(root) / 'val', transform)
        if len(dataset.classes) != 1000:
            raise ValueError('Expected ImageNet val/ with all 1000 class directories.')
    else:
        cls = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        dataset = cls(root, train=False, transform=transform, download=False)
    return dataset, preprocessing


def loader_for(dataset, indices, batch_size, workers):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


class Features(AbstractContextManager):
    """Capture selected module outputs, and always remove hooks on exit."""
    def __init__(self, model, names):
        self.values = {}
        self.handles = []
        for name in names:
            if name == 'norm_final':
                module = model.norm
            elif name.startswith('block_'):
                module = model.blocks[int(name.removeprefix('block_'))]
            elif name.startswith('norm2_'):
                module = model.blocks[int(name.removeprefix('norm2_'))].norm2
            else:
                raise ValueError(f'Unsupported feature layer: {name}')
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def capture(module, inputs, output):
            self.values[name] = output.detach().float()
        return capture

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.values.clear()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
