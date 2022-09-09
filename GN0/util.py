import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph_tool.all import Graph,Vertex
from typing import List, Optional
from torch.utils.tensorboard.writer import SummaryWriter as TorchSummaryWriter
from torch.utils.tensorboard.summary import hparams
import torch
import torch._C
from torch import Tensor
import torch_geometric.utils
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import gather_csr, scatter, segment_csr
from collections import defaultdict

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


class SummaryWriter(TorchSummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)



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
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

