#!/usr/bin/env python3
"""Evaluate an ImageNet checkpoint on val/ class folders."""
import argparse
import json
from pathlib import Path
import torch
from timm import create_model
from timm.data import create_dataset, create_loader
from train import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    model = create_model(args.model, checkpoint_path=args.checkpoint).to(device)
    dataset = create_dataset(root=args.data_dir, split='val')
    if len(dataset.classes) != 1000:
        parser.error('ImageNet val/ must contain all 1000 class folders')
    loader = create_loader(dataset, input_size=(3,224,224), batch_size=args.batch_size,
                           num_workers=args.workers, interpolation='bicubic', device=device)
    print(json.dumps(validate(model, loader, device), indent=2))


if __name__ == '__main__':
    main()
