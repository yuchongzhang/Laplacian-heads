#!/usr/bin/env python3
"""Train DeiT-B on ImageNet class folders with LAMB, BCE, and cosine decay.

Adapted from timm. Copyright 2020 Ross Wightman.
"""
import argparse
import csv
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from timm import create_model
from timm.data import create_dataset, create_loader, FastCollateMixup
from timm.loss import BinaryCrossEntropy
from timm.models import resume_checkpoint
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import AverageMeter, accuracy, random_seed, NativeScaler, ParseKwargs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='')
    parser.add_argument('--data-dir', default='./data/imagenet')
    parser.add_argument('--model', default='deit3_base_12L_patch16_224')
    parser.add_argument('--model-kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size per process')
    parser.add_argument('--validation-batch-size', type=int)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--weight-decay', type=float, default=.02)
    parser.add_argument('--opt-eps', type=float, default=1e-9)
    parser.add_argument('--opt-betas', type=float, nargs=2, default=None)
    parser.add_argument('--clip-grad', type=float, default=1.)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--warmup-lr', type=float, default=1e-6)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--drop', type=float, default=0.)
    parser.add_argument('--drop-path', type=float, default=.4)
    parser.add_argument('--scale', type=float, nargs=2, default=[.08, 1.])
    parser.add_argument('--ratio', type=float, nargs=2, default=[.75, 1.3333333333])
    parser.add_argument('--color-jitter', type=float, default=.3)
    parser.add_argument('--aa', choices=['3a', ''], default='3a')
    parser.add_argument('--aug-repeats', type=int, default=3)
    parser.add_argument('--mixup', type=float, default=.8)
    parser.add_argument('--cutmix', type=float, default=1.)
    parser.add_argument('--smoothing', type=float, default=.1)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--amp-dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--resume', default='')
    parser.add_argument('--output', default='./outputs/imagenet')
    parser.add_argument('--experiment', default='')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--log-wandb', action='store_true')
    parser.add_argument('--wandb-project', default='imagenet_deit3')
    known, _ = parser.parse_known_args(argv)
    if known.config:
        defaults = yaml.safe_load(Path(known.config).read_text()) or {}
        unknown = set(defaults) - {a.dest for a in parser._actions}
        if unknown:
            parser.error(f'Unknown configuration keys: {sorted(unknown)}')
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.epochs, args.log_interval) < 1 or args.workers < 0:
        parser.error('Batch size, epoch count, and log interval must be positive; workers must be nonnegative')
    if (args.mixup or args.cutmix) and args.batch_size % 2:
        parser.error('Mixup requires an even batch size')
    return args


def distributed_device(device_name):
    import os
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if device_name == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    return device, rank, world_size


def reduce_mean(value, world_size):
    if world_size > 1:
        value = value.clone()
        dist.all_reduce(value)
        value /= world_size
    return value


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, device,
                    amp_autocast=nullcontext, loss_scaler=None, rank=0, world_size=1):
    model.train()
    losses = AverageMeter()
    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(loader):
        with amp_autocast():
            loss = loss_fn(model(images), labels)
        if loss_scaler is None:
            loss.backward()
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        else:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
        optimizer.zero_grad()
        losses.update(loss.item(), images.shape[0])
        if rank == 0 and batch_idx % args.log_interval == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] loss {losses.avg:.6f} lr {optimizer.param_groups[0]["lr"]:.6g}', flush=True)
    loss = reduce_mean(torch.tensor(losses.avg, device=device), world_size)
    return {'loss': loss.item()}


@torch.inference_mode()
def validate(model, loader, device, amp_autocast=nullcontext, world_size=1):
    model.eval()
    meters = {name: AverageMeter() for name in ['loss', 'top1', 'top5']}
    for images, labels in loader:
        with amp_autocast():
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        top1, top5 = accuracy(logits, labels, topk=(1, min(5, logits.shape[1])))
        for name, value in zip(meters, [loss, top1, top5]):
            meters[name].update(reduce_mean(value, world_size).item(), images.shape[0])
    return {name: meter.avg for name, meter in meters.items()}


def main():
    from functools import partial
    args = parse_args()
    device, rank, world_size = distributed_device(args.device)
    random_seed(args.seed, rank)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model_options = dict(img_size=args.img_size, num_classes=args.num_classes,
                         drop_rate=args.drop, drop_path_rate=args.drop_path)
    model_options.update(args.model_kwargs)
    model = create_model(args.model, **model_options).to(device)
    optimizer = create_optimizer_v2(model, opt='lamb', lr=args.lr, weight_decay=args.weight_decay,
                                    eps=args.opt_eps, betas=args.opt_betas)
    amp_autocast, loss_scaler = nullcontext, None
    if args.amp and device.type == 'cuda':
        dtype = getattr(torch, args.amp_dtype)
        amp_autocast = partial(torch.autocast, device_type='cuda', dtype=dtype)
        if dtype == torch.float16:
            loss_scaler = NativeScaler()
    start_epoch, best_metric = 0, float('-inf')
    if args.resume:
        start_epoch = resume_checkpoint(model, args.resume, optimizer, loss_scaler)
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        best_metric = saved.get('best_metric', saved.get('metric', float('-inf')))
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
    if not args.no_compile:
        model = torch.compile(model, backend='inductor')
    train_data = create_dataset(root=args.data_dir, split='train')
    val_data = create_dataset(root=args.data_dir, split='val')
    if train_data.class_to_idx != val_data.class_to_idx or len(train_data.classes) != args.num_classes:
        raise ValueError('train/ and val/ must contain the same --num-classes class folders')
    collate = None
    if args.mixup > 0 or args.cutmix > 0:
        collate = FastCollateMixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                                  label_smoothing=args.smoothing, num_classes=args.num_classes)
    loader_options = dict(input_size=(3,args.img_size,args.img_size), num_workers=args.workers,
                          distributed=world_size>1, device=device)
    train_loader = create_loader(train_data, batch_size=args.batch_size, is_training=True,
        num_aug_repeats=args.aug_repeats, collate_fn=collate, interpolation='random',
        auto_augment=args.aa, color_jitter=args.color_jitter, scale=args.scale, ratio=args.ratio,
        **loader_options)
    val_loader = create_loader(val_data, batch_size=args.validation_batch_size or args.batch_size,
                               interpolation='bicubic', **loader_options)
    loss_fn = BinaryCrossEntropy(smoothing=args.smoothing)
    scheduler, _ = create_scheduler_v2(optimizer, num_epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                                      min_lr=args.min_lr, warmup_lr=args.warmup_lr)
    if start_epoch:
        scheduler.step(start_epoch)
    output = Path(args.output)/(args.experiment or f'{datetime.now():%Y%m%d-%H%M%S}-{args.model}')
    wandb = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output/'args.yaml').write_text(yaml.safe_dump(vars(args)))
        if args.log_wandb:
            import wandb
            wandb.init(project=args.wandb_project, name=output.name, config=vars(args))
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, args, device,
                                        amp_autocast, loss_scaler, rank, world_size)
        eval_metrics = validate(model, val_loader, device, amp_autocast, world_size)
        if rank == 0:
            metrics = dict(epoch=epoch, train_loss=train_metrics['loss'],
                           **{f'eval_{k}':v for k,v in eval_metrics.items()}, lr=optimizer.param_groups[0]['lr'])
            summary = output/'summary.csv'
            write_header = not summary.exists()
            with summary.open('a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metrics))
                if write_header:
                    writer.writeheader()
                writer.writerow(metrics)
            if wandb:
                wandb.log(metrics)
            unwrapped = getattr(model, '_orig_mod', model)
            unwrapped = getattr(unwrapped, 'module', unwrapped)
            improved = eval_metrics['top1'] > best_metric
            best_metric = max(best_metric, eval_metrics['top1'])
            saved = dict(epoch=epoch, version=2, state_dict=unwrapped.state_dict(),
                         optimizer=optimizer.state_dict(), best_metric=best_metric, args=vars(args))
            if loss_scaler is not None:
                saved[loss_scaler.state_dict_key] = loss_scaler.state_dict()
            torch.save(saved, output/'last.pth.tar')
            if improved:
                torch.save(saved, output/'model_best.pth.tar')
            print(f'Epoch {epoch}: {eval_metrics}; best top1 {best_metric:.4f}', flush=True)
        scheduler.step(epoch+1)
    if wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
