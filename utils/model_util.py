import torch
from torch import nn
import numpy as np
from math import pi
from einops import rearrange, repeat    # used to reshape, permute, flatten, split, combine or repeat tensor dimensions in a very readable way

def broadcast(tensors, dim=-1):
    """
        broadcast tensors and concatenate along the chosen dimension
    """
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim     # convert the negative index of dimensions to its positive equivalent
    dims = list(zip(*(t.shape for t in tensors)))   # the * unpacks the list so zip works column-wise, final results contain all tensor sizes for a specific dimension index
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]  # identify expandable dimensions: get all dimensions and sizes except the last dimension
    assert all(len(set(val)) == 1 or 1 in val for _, val in expandable_dims), "invalid dimensions for broadcastable concatenation"
    max_dims = [(i, max(val)) for i, val in expandable_dims]    # sizes for broadcastable dimensions
    expanded_dims = [(i, (size,) * num_tensors) for i, size in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))     # now expanded_dims contains all dimensions
    expandable_shapes = list(zip(*(sizes for _, sizes in expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    tensors = [tensor.expand(*shape) for tensor, shape in zip(tensors, expandable_shapes)]  # broadcast the tensor to the target shape
    return torch.cat(tensors, dim = dim)    # concatenate along the chosen dimension

def rotate_half(x):
    pass


class VisionRotaryEmbedding(nn.Module):
    pass

class VisionRotaryEmbeddingFast(nn.Module):
    pass

class RMSNorm(nn.Module):
    pass
