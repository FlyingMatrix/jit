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

