import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear,ModuleList
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean
import copy
from typing import Optional, List, Tuple, Dict, Type, Union, Callable, Any
from torch_geometric.nn.models import GraphSAGE
from torch_scatter.composite import scatter_log_softmax
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.nn import GCNConv, SAGEConv
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, SparseTensor

class ModifiedSAGEConv(SAGEConv):
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """"""
        size = None;
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

class ModifiedGraphSAGE(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.act = activation_resolver("relu")
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm_layer))

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return ModifiedSAGEConv(in_channels, out_channels, aggr="mean", **kwargs).jittable() # aggr sum for detecting num neighbors

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        for i,conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                if self.norms is not None:
                    x = self.norms[i](x)
                if self.act is not None:
                    x = self.act(x)

        return x

class PV_torch_script(torch.nn.Module):
    def __init__(self,hidden_channels,hidden_layers,policy_layers,value_layers,in_channels=3,**gnn_kwargs):
        super().__init__()
        self.gnn = ModifiedGraphSAGE(in_channels=in_channels,hidden_channels=hidden_channels,num_layers=hidden_layers,**gnn_kwargs)

        self.my_modules = torch.nn.ModuleDict()

        self.my_modules["value_head"] = ModifiedGraphSAGE(in_channels=hidden_channels,hidden_channels=hidden_channels,num_layers=value_layers)
        self.my_modules["policy_head"] = ModifiedGraphSAGE(in_channels=hidden_channels,hidden_channels=hidden_channels,num_layers=policy_layers,out_channels=1)

        self.my_modules["value_linear"] = torch.nn.Linear(hidden_channels,1)
        self.my_modules["swap_linear"] = torch.nn.Linear(hidden_channels,1)

        self.value_activation = torch.nn.Tanh()

    def forward(self,x:Tensor,edge_index:Tensor,graph_indices:Tensor,batch_ptr:Tensor):
        assert ((batch_ptr[1:]-batch_ptr[:-1])>2).all() # With only 2 nodes left, someone must have won before
        embeds = self.gnn(x,edge_index)

        pi = self.my_modules["policy_head"](embeds,edge_index)
        value_embeds = self.my_modules["value_head"](embeds,edge_index)
        graph_parts = scatter(value_embeds,graph_indices,dim=0,reduce="sum") # Is mean better?
        value = self.my_modules["value_linear"](graph_parts)
        value = self.value_activation(value)

        should_swap = self.my_modules["swap_linear"](graph_parts)
        should_swap = should_swap.reshape(should_swap.size(0))
        # should_swap = torch.ones_like(should_swap)*100
        # should_swap = self.swap_activation(should_swap)
        pi = pi.reshape(pi.size(0))

        # This part implements the swap rule and removes terminal nodes. It takes up to 5% of total NN time.
        swap_parts = x[batch_ptr[1:-1]-1,2].type(torch.bool)
        swap_indices = batch_ptr[1:-1][swap_parts]
        to_select = torch.ones(pi.size(),dtype=torch.bool, device=x.device)
        to_select[batch_ptr[0]] = False
        to_select[batch_ptr[1:-1]] = swap_parts
        to_select[batch_ptr[:-1]+1] = False

        all_swap_parts = torch.empty(len(batch_ptr),dtype=torch.bool,device=batch_ptr.device)
        all_swap_parts[0] = 0
        all_swap_parts[1:-1] = swap_parts
        all_swap_parts[-1] = x[batch_ptr[-2],2]
        output_batch_ptr = batch_ptr-torch.arange(0,len(batch_ptr)*2,2,device=batch_ptr.device)+torch.cumsum(all_swap_parts,dim=0)

        pi[swap_indices] = should_swap[:-1][swap_parts]
        output_graph_indices = graph_indices.clone()
        output_graph_indices[swap_indices] = output_graph_indices[swap_indices-1]
        pi = pi[to_select]
        output_graph_indices = output_graph_indices[to_select]
        if x[batch_ptr[-2],2]:
            pi = torch.cat((pi,should_swap[-1:]))
            output_graph_indices = torch.cat((output_graph_indices,output_graph_indices[-1:]))

        pi = scatter_log_softmax(pi,index=output_graph_indices)
        return pi,value.reshape(value.size(0)),output_graph_indices,output_batch_ptr
