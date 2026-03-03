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
                    x = [x1, x2, x3, x4, ...]  =>  rotate_half(x) = [-x2, x1, -x4, x3, ...]
    """
    x = rearrange(x, '... (d r) -> ... d r', r=2)   # group every two elements in the last dimension
    x1, x2 = x.unbind(dim=-1)                       # split the last dimension (r=2) into two separate tensors
    x = torch.stack((-x2, x1), dim=-1)              # perform a vectorized 90° rotation of every 2D chunk in parallel
    return rearrange(x, '... d r -> ... (d r)')     # flatten the last two dimensions back into the original shape

class VisionRotaryEmbedding(nn.Module):     # for general rotary embedding -> 2D grid of frequencies
    """
        Vision Rotary Embedding (ViRoPE) is a 2D extension of Rotary Position Embedding (RoPE) used in Transformers.
        It encodes image spatial position (height and width) directly into attention features by rotating query and key vectors in embedding space.
        Mathematically, ViRoPE rotates feature pairs:
            (x1, x2) -> (x1*cos(_theta)-x2*sin(_theta), x1*sin(_theta)+x2*cos(_theta))
        This is a 2D rotation in feature space.
    """
    def __init__(
            self, 
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
            freqs_w = custom_freqs
        elif freqs_for == 'lang':                                       # language mode
            dimension_index = torch.arange(0, dim, 2)[:(dim // 2)]      # RoPE needs: number_of_frequencies = number_of_dimension_pairs = dim//2
            freqs_w = 1. / (theta ** (dimension_index.float() / dim))   # based on classic RoPE frequency formula
        elif freqs_for == 'pixel':                                      # pixel mode, used for vision, linear frequency spacing, better suited for spatial data
            freqs_w = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs_w = torch.ones(num_freqs).float()                     # every dimension uses same frequency
        else:
            raise ValueError(f'unknown modality {freqs_for}')   
        
        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len          # t is the fine-tuning token positions, we stretch them (shorter or longer) into the pretraining coordinate scale

        """
            In standard (1D) RoPE: we rotate features using: theta = position * frequency, then apply: cos(theta) and sin(theta)
            In Vision RoPE (2D case): each token has a height index h and a width index w, so we compute:
                - theta_h = h * frequency
                - theta_w = w * frequency

            Moreover, we define dim//2 frequencies because each frequency governs a 2D rotation pair, 
            we duplicate frequencies into dim, so each pair of embedding channels shares the same frequency.
        """
        # compute height frequency
        freqs_height = t.unsqueeze(-1) * freqs_w                            # (seq_len, dim//2)
        freqs_height = repeat(freqs_height, '... n -> ... (n r)', r = 2)    # (seq_len, dim)

        # compute width frequency
        freqs_width = t.unsqueeze(-1) * freqs_w                             # (seq_len, dim//2)
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

    def forward(self, t, start_index=0):    # apply 2D rotation to pairs of features
        """
            - t: tensor to rotate (e.g., query or key in Attention) -> shape = (batch, height, width, dim*2)
              the last dimension has to match with freqs_cos.shape[-1] = dim*2 to ensure rotation works
            - start_index: where rotation starts in feature dimension
        """
        rot_dim = self.freqs_cos.shape[-1]  # number of dimensions used for rotation
        end_index = start_index + rot_dim
        # the number of dimensions to be rotated should not be larger than the total feature dimension of t
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        # slice feature tensor
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)    # the * is element-wise multiplication
        """ 
            (batch, height, width, dim*2) * (height, width, dim*2) -> (batch, height, width, dim*2)
            (batch, height, width, dim*2) + (batch, height, width, dim*2) = (batch, height, width, dim*2)
            so, t.shape = (batch, height, width, dim*2) -> no dimension changes, only values are rotated 
        """
        return torch.cat((t_left, t, t_right), dim = -1)

class VisionRotaryEmbeddingFast(nn.Module):     # for ViT-style fast inference/training -> standard ViT token layout: [B, N_tokens, D]
    def __init__(
            self, 
            dim,                   # feature dimension to rotate (embedding dimension)
            pt_seq_len=16,         # pretraining sequence length
            ft_seq_len=None,       # fine-tuning sequence length
            custom_freqs=None,     # custom frequency values
            freqs_for='lang',      # mode: 'lang', 'pixel', 'constant'
            theta=10000,           # base scaling constant (default 10000)
            max_freq=10,           # used for pixel frequency scaling
            num_freqs=1,           # used for constant frequency mode
            num_cls_token=0        # number of classification tokens
        ):
        super.__init__()
        """
            num_cls_token tells the module how many special classification tokens are added at the beginning of the sequence, 
            so Rotary Position Embedding (RoPE) can be handled correctly.
            When using a Vision Transformer or BERT-style model, it is usually set to 1.
            When using a pure decoder model without CLS tokens, it's usually 0.        
        """

        if custom_freqs:
            freqs_w = custom_freqs
        elif freqs_for == 'lang':                                       # language mode
            dimension_index = torch.arange(0, dim, 2)[:(dim // 2)]      # RoPE needs: number_of_frequencies = number_of_dimension_pairs = dim//2
            freqs_w = 1. / (theta ** (dimension_index.float() / dim))   # based on classic RoPE frequency formula
        elif freqs_for == 'pixel':                                      # pixel mode, used for vision, linear frequency spacing, better suited for spatial data
            freqs_w = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs_w = torch.ones(num_freqs).float()                     # every dimension uses same frequency
        else:
            raise ValueError(f'unknown modality {freqs_for}')   
        
        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len          # t is the fine-tuning token positions, we stretch them (shorter or longer) into the pretraining coordinate scale
        
        freqs = t.unsqueeze(-1) * freqs                                             # (seq_len, dim//2)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)                          # (seq_len, dim)
        freqs = broad_concat((freqs[:, None, :], freqs[None, :, :]), dim = -1)      # (seq_len, seq_len, dim*2)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # (seq_len*seq_len, dim*2) = (N_img, D)
            cos_img = freqs_flat.cos()                    # (N_img, D)
            sin_img = freqs_flat.sin()                    # (N_img, D)
        
            # add a CLS token to the beginning of the input sequence before feeding it into the model
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)      # (num_cls_token, D)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)     # (num_cls_token, D)
            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()                            # (num_cls_token+N_img, D)
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()                            # (num_cls_token+N_img, D)
        else:
            self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()                           # (N_img, D)
            self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()                           # (N_img, D)

    def forward(self, t):   
        # t.shape = (B, N_tokens, D), where N_tokens = H × W (+ num_cls_token), this matches the standard ViT representation
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin     
        # output.shape = (B, N_tokens, D) -> no dimension changes, only values are rotated

class RMSNorm(nn.Module):   # root mean square normalization
    """
        RMSNorm is a neural network normalization technique that rescales a vector based on its root mean square (RMS) value
        without subtracting the mean.
        It is commonly used in modern transformer models (like LLaMA) as a simpler and faster alternative to LayerNorm.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # create a learnable scaling vector of size hidden_size for model to re-scale each feature dimension appropriately after normalization
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):   # input shape is typically: (batch_size, seq_length, hidden_size), normalization happens across the last dimension
        # save original dtype: models often use float16 or bfloat16, for numerical stability, normalization is done in float32 then converted back
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # compute normalization factor
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # element-wise multiply by learned scaling vector to adjust magnitude per dimension
        return (self.weight * hidden_states).to(input_dtype)
    
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):   
    """
        generate a non-learnable 2D sine-cosine positional embeddings for a square grid of size: grid_size * grid_size
        return shape: (grid_size * grid_size, embed_dim) or (extra_tokens + grid_size * grid_size, embed_dim) when using class token
    """
    # create row and column indices
    grid_h = np.arange(grid_size, dtype=np.float32)     # (grid_size, ) -> 1D array
    grid_w = np.arange(grid_size, dtype=np.float32)     # (grid_size, ) -> 1D array
    # create 2D grid
    grid = np.meshgrid(grid_w, grid_h)                  # grid is a list of 2 arrays, each (grid_size, grid_size)
    grid = np.stack(grid, axis=0)                       # take the list of arrays and stack them along a new axis -> (2, grid_size, grid_size)
    grid = grid.reshape([2, 1, grid_size, grid_size])   # (2, 1, grid_size, grid_size)
    # generate sine-cosine embeddings
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos): 
    """ 
        embed_dim: output dimension for each position
              pos: a list of positions to be encoded -> size (M,)
           output: (M, embed_dim)
    """
    assert embed_dim % 2 == 0                           # sinusoidal encoding splits embed_dim evenly into sine and cosine parts
    i = np.arange(embed_dim//2, dtype=np.float64)       # i = dimension index // 2 -> [0, 1, 2, ..., (embed_dim/2 - 1)]
    omega = 1. / (10000 ** (2 * i / embed_dim))         # (embed_dim/2,)
    pos = pos.reshape(-1)                               # (M,)
    out = np.einsum('m,d->md', pos, omega)              # outer product: out[m,d] = pos[m] ∗ omega[d] -> (M, embed_dim/2)
    emb_sin = np.sin(out)                               # (M, embed_dim/2)
    emb_cos = np.cos(out)                               # (M, embed_dim/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)    # (M, embed_dim): first half -> emb_sin, second half -> emb_cos
    return emb
    
