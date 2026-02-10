import builtins                             # gives access to Python’s built-in functions (like print)
import datetime
import os
import time
from collections import defaultdict, deque  # deque: a fixed-length queue for smoothing metrics, defaultdict: automatically creates metric trackers when a new metric name is seen
from pathlib import Path
import copy

import torch
import torch.distributed as dist            # PyTorch’s distributed training module, this enables multi-GPU and multi-node training

class SmoothedValue(object):
    """
        A utility class to track metrics like loss or accuracy in training loops 
        and provide access to smoothed values over a window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_average:.4f})"
        self.deque = deque(maxlen=window_size)      # stores only the most recent window_size values
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        
    def synchronize_between_processes(self):        # used in distributed training (multiple GPUs/processes)
        if not is_dist_available_and_initialized():
            return
        # pack self.count and self.total into a CUDA tensor
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda') 
        # all processes wait here     
        dist.barrier()  
        # sum t across all processes and distributes the result back, after all_reduce(t), every process holds result.      
        dist.all_reduce(t)    
        # extract synchronized values back into Python variables  
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def average(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_average(self):
        return self.total / self.count
    
    @property
    def max(self):  
        # return the maximum value in the sliding window
        return max(self.deque)
    
    @property
    def value(self):
        # return the latest value added
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            average=self.average,
            global_average=self.global_average,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    """
        A helper class for tracking, smoothing, and printing training metrics
        (like loss, accuracy, time per iteration, etc.) during model training or evaluation.
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)    # a dictionary where every new key automatically gets a SmoothedValue object
        self.delimiter = delimiter
    
    def update(self, **kwargs):    # keyword arguments will be packed into a dictionary
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)    # for self.meters[k], call SmoothedValue.update(v)

    def __getattr__(self, attr):    # logger.attr
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))
    
    def __str__(self):  # print(logger)
        dict = []
        for name, meter in self.meters.items():
            dict.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(dict)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():  # type(meter) -> SmoothedValue
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        pass





def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True