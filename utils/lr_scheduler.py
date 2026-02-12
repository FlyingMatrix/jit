import math

def adjust_lr(optimizer, epoch, args):
    """
        Decay the learning rate with half-cycle cosine after warmup
    """
    if epoch < args.warmup_epochs:   # warmup phase: linearly increase learning rate from 0 -> args.lr
        lr = args.lr * (epoch / args.warmup_epochs)
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            # apply half-cycle cosine decay
            t = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * t))
        else:
            raise NotImplementedError

    for param_group in optimizer.param_groups:  # per-parameter learning rate scaling
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
