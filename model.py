import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils.model_util import VisionRotaryEmbeddingFast, RMSNorm, get_2d_sincos_pos_embed

def modulate(x, shift, scale):
    """
        apply per-feature affine modulation: 
        each feature in x is scaled and shifted independently, allowing the network to adaptively adjust features based on context.

            x: feature tensor from nn, x.shape = (batch_size, seq_len, feature_dim)
        scale: learned or conditional per-feature scaling, scale.shape = (batch_size, feature_dim)
        shift: learned or conditional per-feature shift, shift.shape = (batch_size, feature_dim)

        with element-wise multiplication with broadcasting, output.shape = (batch_size, seq_len, feature_dim)
    """
    output = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return output




