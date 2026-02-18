import math
import sys
import os
import shutil           # shell utilities -> high-level file and directory operations
import torch
import cv2
import copy
import numpy as np
import torch_fidelity   # a pytorch-based toolkit for evaluating generative models, especially GANs and diffusion models
import utils.misc as misc
import utils.lr_scheduler as lr_scheduler

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    """
        train the model for one epoch
    """
    model.train()
    metric_logger = misc.MetricLogger(delimiter="   ")  # delimiter = "\t"
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    optimizer.zero_grad()
    print_freq = 20

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # learning rate scheduler per iteration
        lr_scheduler.adjust_lr(optimizer, epoch + data_iter_step / len(data_loader), args)
        
        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)     # non_blocking=True allows async transfer when using pinned memory
        x = x * 2.0 - 1.0   # from [0, 1] to [-1, 1]
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # enable automatic mixed precision (AMP), run autocast on GPU, use bfloat16 precision for supported ops
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):   # when loss_value == NaN, +inf, -inf
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)     # stop training and handle it

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()    # block the CPU until all queued CUDA operations on the GPU are finished
        model_without_ddp.update_ema()  # update the EMA of the model parameters to smooth or stabilize the model's weights over training steps
        
        # add loss_value and lr into deque (double-ended queue)
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # calculate mean value across multiple GPUs (processes)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            pass






        

    



def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    """
        generate images and computes FID / IS for evaluation
            FID (Fréchet Inception Distance) compares real images vs generated images in feature space, it measures:
                - image quality
                - diversity
                - similarity to real data distribution
            IS (Inception Score) evaluates image quality and diversity
    """
    pass

