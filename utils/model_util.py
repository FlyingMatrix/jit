import torch
from torch import nn
import numpy as np
from math import pi
from einops import rearrange, repeat    # used to reshape, permute, flatten, split, combine or repeat tensor dimensions in a very readable way

def broad_concat(tensors, dim=-1):
    """
        broadcast tensors and concatenate along the chosen dimension
    """
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim                             # convert the negative index of dimensions to its positive equivalent
    dims = list(zip(*(t.shape for t in tensors)))                           # the * unpacks the list so zip works column-wise, final results contain all tensor sizes for a specific dimension index
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]  # identify expandable dimensions: get all dimensions and sizes except the last dimension
    assert all(len(set(val)) == 1 or 1 in val for _, val in expandable_dims), "invalid dimensions for broadcastable concatenation"
    max_dims = [(i, max(val)) for i, val in expandable_dims]                # sizes for broadcastable dimensions
    expanded_dims = [(i, (size,) * num_tensors) for i, size in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))                             # now expanded_dims contains all dimensions
    expandable_shapes = list(zip(*(sizes for _, sizes in expanded_dims)))
    tensors = [tensor.expand(*shape) for tensor, shape in zip(tensors, expandable_shapes)]  # broadcast the tensor to the target shape
    return torch.cat(tensors, dim=dim)                                      # concatenate along the chosen dimension

def rotate_half(x):
    """
        take a tensor whose last dimension is even, splits it into pairs, rotates each pair by 90° in 2D, and flattens back
        common use: Rotary Positional Embeddings (RoPE) in Transformers
    """
    x = rearrange(x, '... (d r) -> ... d r', r=2)   # group every two elements in the last dimension
    x1, x2 = x.unbind(dim=-1)                       # split the last dimension (r=2) into two separate tensors
    x = torch.stack((-x2, x1), dim=-1)              # perform a vectorized 90° rotation of every 2D chunk in parallel
    return rearrange(x, '... d r -> ... (d r)')     # flatten the last two dimensions back into the original shape

class VisionRotaryEmbedding(nn.Module):
    """
        Vision Rotary Embedding (ViRoPE) is a 2D extension of Rotary Position Embedding (RoPE) used in Transformers.
        It encodes image spatial position (height and width) directly into attention features by rotating query and key vectors in embedding space.
        Mathematically, RoPE rotates feature pairs:
            (x1, x2) -> (x1*cos(_theta)-x2*sin(_theta), x1*sin(_theta)+x2*cos(_theta))
        This is a 2D rotation in feature space.
    """
    def __init__(self, 
                 dim,                   # feature dimension to rotate (embedding dimension)
                 pt_seq_len,            # pretraining sequence length
                 ft_seq_len=None,       # fine-tuning sequence length
                 custom_freqs=None,     # custom frequency values
                 freqs_for='lang',      # mode: 'lang', 'pixel', 'constant'
                 theta=10000,           # base scaling constant (default 10000)
                 max_freq=10,           # used for pixel frequency scaling
                 num_freqs=1            # used for constant frequency mode
                ):
        super.__init__()

        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':                                       # language mode
            dimension_index = torch.arange(0, dim, 2)[:(dim // 2)]
            freqs_w = 1. / (theta ** (dimension_index.float() / dim))   # based on classic RoPE frequency formula
        elif freqs_for == 'pixel':                                      # pixel mode, used for vision, linear frequency spacing, better suited for spatial data
            freqs_w = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs_w = torch.ones(num_freqs).float()                     # every dimension uses same frequency
        else:
            raise ValueError(f'unknown modality {freqs_for}')   
        
        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len          # mapping sequence indices from fine-tuning space into the pre-training space

        """
            In standard (1D) RoPE: we rotate features using: theta = position * frequency, then apply: cos(theta) and sin(theta)
            In Vision RoPE (2D case): each token has a height index h and a width index w, so we compute:
                - theta_h = h * frequency
                - theta_w = w * frequency

            Moreover, we define dim // 2 frequencies because each frequency governs a 2D rotation pair, 
            and we duplicate them so each pair of embedding channels shares the same frequency.
        """
        # compute height frequency
        freqs_height = t.unsqueeze(-1) * freqs_w                            # (seq_len, dim // 2)
        freqs_height = repeat(freqs_height, '... n -> ... (n r)', r = 2)    # (seq_len, dim)

        # compute width frequency
        freqs_width = t.unsqueeze(-1) * freqs_w                             # (seq_len, dim // 2)
        freqs_width = repeat(freqs_width, '... n -> ... (n r)', r = 2)      # (seq_len, dim)

        # broadcast and concatenate freqs_height with freqs_width
        freqs = broad_concat((freqs_height[:, None, :], freqs_width[None, :, :]), dim=-1)  
        """
            # freqs_height[:, None, :] -> (height, 1, dim)
            # freqs_width[None, :, :] -> (1, width, dim)
            # expandable_shapes -> [(height, width, dim), (height, width, dim)]
            # freqs -> (height, width, dim*2)
        """

        # store precomputed cosine and sine rotation factors inside the module which can be automatically moved to GPU (not trainable)
        self.register_buffer("freqs_cos", freqs.cos())      # (height, width, dim*2)
        self.register_buffer("freqs_sin", freqs.sin())      # (height, width, dim*2)

    def forward(self, t, start_index=0):    # apply rotation
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        


class VisionRotaryEmbeddingFast(nn.Module):
    pass

class RMSNorm(nn.Module):
    pass
