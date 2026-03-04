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

class BottleneckPatchEmbedder(nn.Module):  # image to patch embedding
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
    
class TimestepEmbedder(nn.Module):  # scalar timesteps embedding
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super.__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
             input: a tensor of timesteps t of shape (batch_size,)
            output: an embedding vector of shape (batch_size, dim)
        """
        # calculate frequencies based on the function: freqs = max_period ** (-index / half), which is from openai -> glide_text2im/nn.py
        half = dim // 2
        index = torch.arange(start=0, end=half, dtype=torch.float32)     # index -> (dim//2,)
        freqs = max_period ** (-index / half).to(device=t.device)        # freqs -> (dim//2,)
        # multiply by timesteps
        args = t[:, None].float() * freqs[None]     # (batch_size, 1) * (1, dim//2) -> (batch_size, dim//2)
        # concatenate
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # embedding.shape = torch.Tensor(batch_size, dim)
        # in odd dimension case, one zero column is appended
        if dim % 2: # in odd dimension case, embedding.shape = torch.Tensor(batch_size, dim-1)
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # now, embedding.shape = torch.Tensor(batch_size, dim)
        return embedding    # embedding.shape = torch.Tensor(batch_size, dim)
    
    def forward(self, t):   # t.shape = (batch_size,)
        t_embedding = self.timestep_embedding(t, self.frequency_embedding_size)  # t_embedding.shape = torch.Tensor(batch_size, self.frequency_embedding_size)
        # pass the t_embedding through MLP to make it can be learned by the model
        t_embedding = self.mlp(t_embedding)
        return t_embedding  # t_embedding.shape = torch.Tensor(batch_size, hidden_size)


