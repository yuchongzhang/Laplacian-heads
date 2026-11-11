#!/usr/bin/env python3
"""CIFAR Training

Train DeiT-B on CIFAR-10 or CIFAR-100.
Designed for single-GPU training.
"""
import argparse
import json
import logging
import os
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml

from timm import utils
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger('train_cifar')

DATASETS = {
    'cifar10': (10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': (100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

def get_args_parser(dataset):
    parser = argparse.ArgumentParser(description=f'{dataset.upper()} DeiT-B training')

    # Config file
    parser.add_argument('-c', '--config', default='', type=str, help='YAML config file')

    # Model
    parser.add_argument('--model', default='deit3_base_qk_norm_patch4_32', type=str, help='Model name from timm')
    parser.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs,
                        help='Model keyword arguments')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--opt-betas', type=float, nargs='+', default=None, help='Optimizer betas')
    # Scheduler
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum LR')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='Warmup LR')

    # Data
    parser.add_argument('--data-dir', default='./data', type=str, help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Data loading workers')

    # Augmentation
    parser.add_argument('--aa', choices=['', 'rand-m9-mstd0.5'], default='rand-m9-mstd0.5', help='RandAugment policy; empty disables')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0.0, help='Cutmix alpha')
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing')

    # Regularization
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--drop-path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient clipping norm')

    # Training
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--ckpt-mode', default='full', choices=['full', 'best', 'none'],
                        help="Checkpointing: 'full' = last.pth every epoch + best.pth, both with "
                             "optimizer state (~2GB/run, resumable); 'best' = best.pth only, model "
                             "weights without optimizer state (~340MB/run, not resumable); "
                             "'none' = no .pth files. Metrics are recorded "
                             "in training_log.json regardless.")

    # Logging
    parser.add_argument('--output', default=f'./outputs/{dataset}', type=str, help='Output directory')
    parser.add_argument('--experiment', default='', type=str, help='Experiment name')
    parser.add_argument('--log-interval', type=int, default=50, help='Log interval')
    parser.add_argument('--log-wandb', action='store_true', help='Log to wandb')
    parser.add_argument('--wandb-project', default=dataset, type=str, help='Wandb project')

    return parser


def parse_args(argv=None, dataset='cifar100'):
    parser = get_args_parser(dataset)
    known, _ = parser.parse_known_args(argv)
    if known.config:
        with open(known.config) as handle:
            defaults = yaml.safe_load(handle) or {}
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.dataset = dataset
    return args


def create_dataloaders(args):
    _, mean, std = DATASETS[args.dataset]
    # Training transforms - use CIFAR-appropriate augmentation
    from timm.data.auto_augment import rand_augment_transform

    train_transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    # Apply RandAugment before conversion to a normalized tensor.
    if args.aa:
        aa_params = dict(
            translate_const=int(32 * 0.45),  # CIFAR-appropriate translation
            img_mean=tuple([int(x * 255) for x in mean]),
        )
        train_transform_list.append(rand_augment_transform(args.aa, aa_params))

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset_cls = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100

    train_dataset = dataset_cls(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = dataset_cls(
        root=args.data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None,
    )

    return train_loader, val_loader


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, device, args,
                    lr_scheduler=None, amp_autocast=suppress, loss_scaler=None, mixup_fn=None):
    model.train()
    losses = utils.AverageMeter()

    num_updates = epoch * len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, memory_format=torch.channels_last)
        targets = targets.to(device)

        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        with amp_autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
        else:
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        num_updates += 1

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates)

        if batch_idx % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            _logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(loader)}] '
                        f'Loss: {losses.val:.4f} ({losses.avg:.4f}) LR: {lr:.6f}')

    return OrderedDict([('loss', losses.avg)])


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp_autocast=suppress):
    model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for inputs, targets in loader:
        inputs = inputs.to(device, memory_format=torch.channels_last)
        targets = targets.to(device)

        with amp_autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

    return OrderedDict([('loss', losses.avg), ('top1', top1.avg), ('top5', top5.avg)])


def main(dataset='cifar100'):
    args = parse_args(dataset=dataset)
    num_classes = DATASETS[dataset][0]

    # Setup
    utils.random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Create model
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        **args.model_kwargs,
    )
    model = model.to(device, memory_format=torch.channels_last)
    _logger.info(f'Model {args.model} created, params: {sum(p.numel() for p in model.parameters()):,}')

    # Group parameters before compilation changes their names.
    opt_kwargs = dict(
        opt="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.opt_betas is not None:
        opt_kwargs['betas'] = tuple(args.opt_betas)

    optimizer = create_optimizer_v2(model, **opt_kwargs)

    # Compile model for faster training (PyTorch 2.0+)
    if hasattr(torch, 'compile') and not args.no_compile:
        model = torch.compile(model)
        _logger.info('Model compiled with torch.compile()')

    # AMP
    amp_autocast = suppress
    loss_scaler = None
    if args.amp and device.type == 'cuda':
        amp_autocast = partial(torch.autocast, device_type='cuda', dtype=torch.float16)
        loss_scaler = NativeScaler()
        _logger.info('Using mixed precision training')

    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        from timm.models._helpers import clean_state_dict
        unwrapped = getattr(model, '_orig_mod', model)
        unwrapped.load_state_dict(clean_state_dict(checkpoint['model']))
        if loss_scaler is not None and 'amp_scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['amp_scaler'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        _logger.info(f'Resumed from epoch {start_epoch}')

    # Data
    train_loader, val_loader = create_dataloaders(args)

    # Mixup / Cutmix
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            label_smoothing=args.smoothing,
            num_classes=num_classes,
        )

    # Loss
    if mixup_fn is not None:
        train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device)
    val_loss_fn = nn.CrossEntropyLoss().to(device)

    # Scheduler
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        updates_per_epoch=len(train_loader),
    )

    if start_epoch:
        lr_scheduler.step(start_epoch)

    # Output directory
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{safe_model_name(args.model)}'
    output_dir = os.path.join(args.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Wandb
    if args.log_wandb and not has_wandb:
        raise RuntimeError('Install wandb to use --log-wandb')
    if args.log_wandb:
        wandb.init(project=args.wandb_project, name=exp_name, config=vars(args))

    # JSON logging
    json_log_path = os.path.join(output_dir, 'training_log.json')
    training_history = {
        'config': vars(args),
        'model_params': sum(p.numel() for p in model.parameters()),
        'epochs': []
    }
    # Load existing history if resuming
    if args.resume and os.path.exists(json_log_path):
        with open(json_log_path, 'r') as f:
            training_history = json.load(f)

    # Training loop
    best_metric = float(checkpoint.get('best_metric', 0)) if args.resume else 0
    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_one_epoch(
            epoch, model, train_loader, optimizer, train_loss_fn, device, args,
            lr_scheduler=lr_scheduler, amp_autocast=amp_autocast,
            loss_scaler=loss_scaler, mixup_fn=mixup_fn
        )

        val_metrics = validate(model, val_loader, val_loss_fn, device, amp_autocast)

        _logger.info(f'Epoch {epoch}: Train Loss: {train_metrics["loss"]:.4f}, '
                    f'Val Loss: {val_metrics["loss"]:.4f}, '
                    f'Val Top1: {val_metrics["top1"]:.2f}%, Val Top5: {val_metrics["top5"]:.2f}%')

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        if args.log_wandb and has_wandb:
            log_data = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_top1': val_metrics['top1'],
                'val_top5': val_metrics['top5'],
                'lr': current_lr,
            }
            wandb.log(log_data)

        is_best = val_metrics['top1'] > best_metric
        best_metric = max(val_metrics['top1'], best_metric)

        # JSON logging
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_top1': val_metrics['top1'],
            'val_top5': val_metrics['top5'],
            'lr': current_lr,
        }
        training_history['epochs'].append(epoch_data)
        if is_best:
            training_history['best_epoch'] = epoch
        with open(json_log_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        # Save checkpoint
        if args.ckpt_mode == 'full':
            checkpoint = {
                'epoch': epoch,
                'model': getattr(model, '_orig_mod', model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_metric,
                'args': vars(args),
            }
            if loss_scaler is not None:
                checkpoint['amp_scaler'] = loss_scaler.state_dict()
            torch.save(checkpoint, os.path.join(output_dir, 'last.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))
        elif args.ckpt_mode == 'best' and is_best:
            # Model weights only - no optimizer state, so ~1/3 the size and not resumable.
            torch.save({
                'epoch': epoch,
                'model': getattr(model, '_orig_mod', model).state_dict(),
                'best_metric': best_metric,
                'args': vars(args),
            }, os.path.join(output_dir, 'best.pth'))

    _logger.info(f'Training complete. Best Top-1: {best_metric:.2f}%')

    # Final JSON update with summary
    training_history['best_top1'] = best_metric
    training_history['total_epochs'] = num_epochs
    with open(json_log_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    if args.log_wandb and has_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
