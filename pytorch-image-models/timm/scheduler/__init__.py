from .cosine_lr import CosineLRScheduler

def create_scheduler_v2(optimizer, sched='cosine', num_epochs=300, warmup_epochs=0,
                        min_lr=0., warmup_lr=1e-5, updates_per_epoch=0):
    if sched != 'cosine':
        raise ValueError('Only cosine scheduling is supported')
    scheduler = CosineLRScheduler(optimizer, num_epochs, lr_min=min_lr,
                                  warmup_t=warmup_epochs, warmup_lr_init=warmup_lr)
    return scheduler, num_epochs
