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

class BottleneckPatchEmbed(nn.Module):  # image to patch embedding
    def __init__(self, img_size=224, patch_size=16, in_channels=3, pca_dim=768, embed_dim=768, bias=True):
        """
            - img_size: size of input image
            - patch_size: size of each patch 
            - in_channels: number of channels in the input image
            - pca_dim: intermediate channel dimension (bottleneck dimension)
            - embed_dim: final embedding dimension for each patch
            - bias: if the final convolution has a bias term
        """
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_channels, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x))       # x.shape = (batch_size, embed_dim, img_size//patch_size, img_size//patch_size)
        x = x.flatten(2).transpose(1, 2)    # x.shape = (batch_size, num_patches, embed_dim)
        return x    
    




