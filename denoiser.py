import torch 
import torch.nn as nn

class Denoiser(nn.Module):
    def __init__(self, args):
        pass

    @torch.no_grad()
    def update_ema(self):
        pass

    @torch.no_grad()
    def generate(self, labels):
        pass
