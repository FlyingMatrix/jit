import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn                    # access to NVIDIA’s cuDNN backend for GPU-accelerated deep learning operations
from torch.utils.tensorboard import SummaryWriter       # TensorBoard logging for PyTorch 
import torchvision.transforms as transforms             # preprocess images, e.g., normalization, resizing, cropping, flipping, etc.
import torchvision.datasets as datasets                 # access to datasets

from utils.crop import center_crop_arr
import utils.misc as misc

import copy


def get_args_parser():
    pass

def main(args):
    pass

if __name__ == '__main__':
    pass