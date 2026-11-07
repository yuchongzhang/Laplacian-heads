"""Cosine decay with linear warmup. Copyright 2021 Ross Wightman."""
import math

class CosineLRScheduler:
    def __init__(self, optimizer, t_initial, lr_min=0., warmup_t=0, warmup_lr_init=0.):
        self.optimizer = optimizer
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_values = [group['initial_lr'] for group in optimizer.param_groups]
        self.warmup_steps = [(v - warmup_lr_init) / warmup_t for v in self.base_values] if warmup_t else []
        if warmup_t:
            self._update([warmup_lr_init] * len(self.base_values))

    def _update(self, values):
        for group, value in zip(self.optimizer.param_groups, values):
            group['lr'] = value

    def step(self, epoch, metric=None):
        if epoch < self.warmup_t:
            values = [self.warmup_lr_init + epoch * step for step in self.warmup_steps]
        elif epoch < self.t_initial:
            values = [self.lr_min + .5 * (v - self.lr_min) * (1 + math.cos(math.pi * epoch ** 1. / self.t_initial ** 1.)) for v in self.base_values]
        else:
            values = [self.lr_min] * len(self.base_values)
        self._update(values)

    def step_update(self, num_updates, metric=None):
        pass  # This schedule advances once per epoch.
