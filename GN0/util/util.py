"""Some unorganized neural network utilities used in the python algorithms"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from graph_tool.all import Graph,Vertex
from typing import List, Optional
import torch
import torch._C
from torch import Tensor
import torch_geometric.utils
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import gather_csr, scatter, segment_csr, scatter_add, scatter_max
from collections import defaultdict

def downsample_cnn_outputs(q_values,target_hex_size):
    if len(q_values.shape) == 2:
        if q_values.shape[1] == target_hex_size**2:
            return q_values
        q_size = int(sqrt(q_values.shape[1]))
        return q_values.reshape(q_values.shape[0],q_size,q_size)[:,:target_hex_size,:target_hex_size].reshape(q_values.shape[0],-1)
    elif len(q_values.shape) == 1:
        if len(q_values) == target_hex_size**2:
            return q_values
        q_size = int(sqrt(len(q_values)))
        return q_values.reshape(q_size,q_size)[:target_hex_size,:target_hex_size].flatten()
    else:
        print(q_values.shape)
        raise ValueError()

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class fix_size_defaultdict(defaultdict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        defaultdict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.pop(next(iter(self)))

class Identity():
    def __init__(self,*args,**kwargs):
        pass
    def __call__(self,x):
        return x

def log_softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated log_softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the log_softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        minusmax = src-src_max
        out = minusmax.exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        minusmax = src-src_max
        out = minusmax.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return minusmax - torch.log(out_sum + 1e-16)

def graph_NLLLoss(
    pred: Tensor,
    targets: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0
) -> Tensor:
    r"""Computes the negative log likelihood loss for a graph.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`
    """
    if ptr is not None:
        dim = dim + pred.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        summed = gather_csr(segment_csr(-pred*targets,ptr,reduce='sum'),ptr)
        res = torch.mean(summed)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        summed = scatter(-pred*targets,index,dim_size=N, reduce="sum")
        res = torch.mean(summed)
    else:
        raise NotImplementedError
    return res

def graph_cross_entropy(pred: Tensor, targets:Tensor, index:Optional[Tensor] = None, ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None, dim: int = 0):
    pred = torch.log(torch_geometric.utils.softmax(pred,index=index,ptr=ptr,num_nodes=num_nodes,dim=dim))
    return graph_NLLLoss(pred,targets,index=index,ptr=ptr,num_nodes=num_nodes,dim=dim)


def get_one_hot(length:int,index:int,dtype=np.float32):
    """Returns a zero vector with one entry set to one
    
    Args:
        index: The index of the entry to set to one
        length: The lenght of the output vector
        dtype: The dtype of the desired vector

    Returns:
        A numpy array of one-hot format
    """
    b = np.zeros(length,dtype=np.float32)
    b[index] = 1
    return b

def get_alternating(length:int,even,odd,dtype=np.float32):
    """Get an array with alternating values

    Args:
        length: The length of the desired array
        even: The value to put at even indices of the array (Assuming 0 as starting index)
        odd: The value to put at odd indices of the array
        dtype: The dtype of the desired array

    Returns:
        A numpy array with alternating values
    """
    out = np.empty(length,dtype=dtype)
    out[::2] = even
    out[1::2] = odd
    return out

def visualize_graph(G, color): # SOURCE https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
    """Visualize a graph with networkx and matplotlib
    
    Args:
        G: The graph to visualize
        color: The color of the graph nodes
    """
    import networkx as nx
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

def count_model_parameters(model:torch.nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
