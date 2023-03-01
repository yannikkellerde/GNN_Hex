import torch
from torch import Tensor
from typing import Optional, List, Tuple, Dict, Type, Union, Callable, Any
import GN0.unet_parts as unet

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import GCNConv, SAGEConv, PNAConv
from torch_geometric.data import Batch, Data
from torch_scatter import scatter, scatter_mean
from torch_scatter.composite import scatter_log_softmax
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, SparseTensor
from torch_geometric.nn.aggr.scaler import DegreeScalerAggregation
from torch_geometric.nn.models import PNA
from torch.nn import Linear,ModuleList
import torch.nn.functional as F
import copy
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from rl_loop.old_models import PV_torch_script


class ModifiedPNAConv(PNAConv):

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        h = torch.cat([x_i, x_j], dim=-1)
        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)


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

class ModifiedBaseNet(torch.nn.Module):
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
        conv_class = ModifiedSAGEConv,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.conv_class = conv_class
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
        return self.conv_class(in_channels, out_channels, **kwargs).jittable() # aggr sum for detecting num neighbors

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
        # This is obviously terribly redundant terrible code. But it is jit tracable.
        if self.norms is None:
            for i,conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i != self.num_layers - 1:
                    if self.act is not None:
                        x = self.act(x)
        else:
            assert len(self.norms)==len(self.convs)-1
            for i,(norm,conv) in enumerate(zip(self.norms,self.convs)):
                x = conv(x, edge_index)
                if i != self.num_layers - 1:
                    x = norm(x)
                    if self.act is not None:
                        x = self.act(x)
            x = self.convs[-1](x,edge_index)

        return x

# Statistics measured for hex 7x7
deg_hist = torch.tensor([40,88,3444,8054,6863,3415,8412,1737,1205,300,100,4],dtype=torch.long)
aggregators = ['mean', 'min', 'max']
scalers = ['identity', 'amplification', 'attenuation']

# aggregators = ['mean','max','min']
# scalers = ['attenuation']

class PNA_torch_script(torch.nn.Module):
    def __init__(self,hidden_channels,hidden_layers,policy_layers,value_layers,in_channels=3):
        super().__init__()
        self.gnn = ModifiedBaseNet(in_channels=in_channels,norm=None,hidden_channels=hidden_channels,num_layers=hidden_layers,conv_class=ModifiedPNAConv,deg=deg_hist,aggregators=aggregators, scalers=scalers)

        self.my_modules = torch.nn.ModuleDict()

        self.my_modules["value_head"] = ModifiedBaseNet(in_channels=hidden_channels,norm=None,hidden_channels=hidden_channels,num_layers=value_layers,conv_class=ModifiedPNAConv,deg=deg_hist,aggregators=aggregators,scalers=scalers)
        self.my_modules["policy_head"] = ModifiedBaseNet(in_channels=hidden_channels,norm=None,hidden_channels=hidden_channels,num_layers=policy_layers,out_channels=1,conv_class=ModifiedPNAConv,deg=deg_hist,aggregators=aggregators,scalers=scalers)


        # self.pre_head_layer_norm = LayerNorm(hidden_channels)

        self.value_activation = torch.nn.Tanh()

        self.my_modules["value_linear"] = MLP(hidden_channels//2,1,hidden_channels*4,1,output_activation=self.value_activation)
        self.my_modules["swap_linear"] = MLP(hidden_channels//2,1,hidden_channels*4,1)


    def forward(self,x:Tensor,edge_index:Tensor,graph_indices:Tensor,batch_ptr:Tensor):
        assert ((batch_ptr[1:]-batch_ptr[:-1])>2).all() # With only 2 nodes left, someone must have won before
        # embeds = self.pre_head_layer_norm(self.gnn(x,edge_index))
        embeds = self.gnn(x,edge_index)

        pi = self.my_modules["policy_head"](embeds,edge_index)
        value_embeds = self.my_modules["value_head"](embeds,edge_index)
        graph_parts_sum = scatter(value_embeds,graph_indices,dim=0,reduce="sum")
        graph_parts_max = scatter(value_embeds,graph_indices,dim=0,reduce="max")
        graph_parts_min = scatter(value_embeds,graph_indices,dim=0,reduce="min")
        graph_parts_mean = scatter(value_embeds,graph_indices,dim=0,reduce="mean")
        graph_parts = torch.cat([graph_parts_sum,graph_parts_max,graph_parts_min,graph_parts_mean],dim=1)
        value = self.my_modules["value_linear"](graph_parts)

        should_swap = self.my_modules["swap_linear"](graph_parts)
        should_swap = should_swap.reshape(should_swap.size(0))
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

class MLP(torch.nn.Module):
    def __init__(self,hidden_channels,num_hidden_layers,num_input,num_output,output_activation=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if num_hidden_layers==0:
            self.layers.append(torch.nn.Linear(num_input,num_output))
        else:
            self.layers.append(torch.nn.Linear(num_input,hidden_channels))
            for i in range(num_hidden_layers-1):
                self.layers.append(torch.nn.Linear(hidden_channels,hidden_channels))
            self.layers.append(torch.nn.Linear(hidden_channels,num_output))
        self.output_activation = output_activation

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            if layer is not self.layers[-1]:
                x = F.relu(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
        

class SAGE_torch_script(torch.nn.Module):
    def __init__(self,hidden_channels,hidden_layers,policy_layers,value_layers,in_channels=3,swap_allowed=False,norm=None,**gnn_kwargs):
        super().__init__()
        self.swap_allowed = swap_allowed
        self.gnn = ModifiedBaseNet(in_channels=in_channels,norm=None if norm is None else norm(hidden_channels),hidden_channels=hidden_channels,num_layers=hidden_layers,conv_class=ModifiedSAGEConv,**gnn_kwargs)

        self.my_modules = torch.nn.ModuleDict()

        self.my_modules["value_head"] = ModifiedBaseNet(in_channels=hidden_channels,norm=None if norm is None else norm(hidden_channels),hidden_channels=hidden_channels,conv_class=ModifiedSAGEConv,num_layers=value_layers)
        self.my_modules["policy_head"] = ModifiedBaseNet(in_channels=hidden_channels,norm=None if norm is None else norm(hidden_channels),hidden_channels=hidden_channels,num_layers=policy_layers,conv_class=ModifiedSAGEConv,out_channels=1)

        # self.my_modules["value_linear"] = torch.nn.Linear(hidden_channels*4,1)
        # self.my_modules["swap_linear"] = torch.nn.Linear(hidden_channels*4,1)
        self.my_modules["value_linear"] = MLP(hidden_channels//2,1,hidden_channels*4,1)
        # if self.swap_allowed:
        self.my_modules["swap_linear"] = MLP(hidden_channels//2,1,hidden_channels*4,1)
    
        self.before_head_norm = None if norm is None else norm(hidden_channels)

        self.value_activation = torch.nn.Tanh()

    def forward(self,x:Tensor,edge_index:Tensor,graph_indices:Tensor,batch_ptr:Tensor):
        assert ((batch_ptr[1:]-batch_ptr[:-1])>2).all() # With only 2 nodes left, someone must have won before
        embeds = self.gnn(x,edge_index)
        if self.before_head_norm is not None:
            embeds = self.before_head_norm(embeds)

        pi = self.my_modules["policy_head"](embeds,edge_index)
        value_embeds = self.my_modules["value_head"](embeds,edge_index)
        graph_parts_sum = scatter(value_embeds,graph_indices,dim=0,reduce="sum") 
        graph_parts_max = scatter(value_embeds,graph_indices,dim=0,reduce="max")
        graph_parts_min = scatter(value_embeds,graph_indices,dim=0,reduce="min")
        graph_parts_mean = scatter(value_embeds,graph_indices,dim=0,reduce="mean")
        graph_parts = torch.cat([graph_parts_sum,graph_parts_max,graph_parts_min,graph_parts_mean],dim=1)

        value = self.my_modules["value_linear"](graph_parts)
        value = self.value_activation(value)

        pi = pi.reshape(pi.size(0))

        # This part implements the swap rule and removes terminal nodes. It takes up to 5% of total NN time.
        # Swap probabilities are inserted at the index of the first terminal node in the next graph.
        # Then all indices for terminal nodes are removed (except the ones where the swap was inserted)
        # Terminal nodes are the first two for each graph.
        if self.swap_allowed:
            should_swap = self.my_modules["swap_linear"](graph_parts)
            should_swap = should_swap.reshape(should_swap.size(0))
            swap_parts = x[batch_ptr[1:-1]-1,2].type(torch.bool)  # indicates if swapping is allowed in each graph
            swap_indices = batch_ptr[1:-1][swap_parts]  # At these indices, the swap probabilities will be inserted
            to_select = torch.ones(pi.size(),dtype=torch.bool, device=x.device)  # Parts of the policy output to be kept
            to_select[batch_ptr[0]] = False # Do not keep first terminal node of first graph
            to_select[batch_ptr[1:-1]] = swap_parts # Keep slot of first terminal node for other graphs, only if swap is allowed in the previous graph
            to_select[batch_ptr[:-1]+1] = False # Do not keep second terminal node slots
            all_swap_parts = torch.empty(len(batch_ptr),dtype=torch.bool,device=batch_ptr.device) # This is just the swap_parts with 2 extra length for first and last index
            all_swap_parts[0] = 0
            all_swap_parts[1:-1] = swap_parts
            all_swap_parts[-1] = x[batch_ptr[-2],2]
            # Compute the new batch pointer after removing the terminal nodes. Use cumsum on all_swap_parts to account for the indices that are kept for swap probabilities.
            output_batch_ptr = batch_ptr-torch.arange(0,len(batch_ptr)*2,2,device=batch_ptr.device)+torch.cumsum(all_swap_parts,dim=0)
            # Fill the swap indices in the policy with the swap probabilities
            pi[swap_indices] = should_swap[:-1][swap_parts]
            output_graph_indices = graph_indices.clone()
            # In output graph_indices, the swap parts belong to the graph before
            output_graph_indices[swap_indices] = output_graph_indices[swap_indices-1]
            # Finally remove unwanted terminal nodes from pi
            pi = pi[to_select]
            # ... And from graph_indices
            output_graph_indices = output_graph_indices[to_select]
            # We did not account for the swap probabilities of the last graph in the batch. So we need to add it here.
            if x[batch_ptr[-2],2]:
                pi = torch.cat((pi,should_swap[-1:]))
                output_graph_indices = torch.cat((output_graph_indices,output_graph_indices[-1:]))
        else:
            to_select = torch.ones(pi.size(),dtype=torch.bool, device=x.device)  # Parts of the policy output to be kept
            to_select[batch_ptr[:-1]] = False
            to_select[batch_ptr[:-1]+1] = False # Do not keep second terminal node slots
            output_batch_ptr = batch_ptr-torch.arange(0,len(batch_ptr)*2,2,device=batch_ptr.device)
            
            output_graph_indices = graph_indices.clone()

            # Finally remove unwanted terminal nodes from pi
            pi = pi[to_select]
            # ... And from graph_indices
            output_graph_indices = output_graph_indices[to_select]

        pi = scatter_log_softmax(pi,index=output_graph_indices)
        return pi,value.reshape(value.size(0)),output_graph_indices,output_batch_ptr

class Unet(torch.nn.Module):
    def __init__(self,in_channels,starting_channels=9):
        super().__init__()
        self.inc = (unet.DoubleConv(in_channels, starting_channels))
        self.down1 = (unet.Down(starting_channels, starting_channels*2))
        self.down2 = (unet.Down(starting_channels*2, starting_channels*4))
        self.down3 = (unet.Down(starting_channels*4, starting_channels*8))
        self.up1 = (unet.Up(starting_channels*8, starting_channels*4))
        self.up2 = (unet.Up(starting_channels*4, starting_channels*2))
        self.up3 = (unet.Up(starting_channels*2, starting_channels))
        self.out_policy = (unet.OutConv(starting_channels, 1))
        self.out_value = (unet.OutConv(starting_channels, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        policy = self.out_policy(x)
        policy = policy.reshape((policy.shape[0],policy.shape[1],policy.shape[2]*policy.shape[3]))
        policy = F.log_softmax(policy,dim=2)
        value = self.out_value(x)
        value = torch.mean(value,(2,3))
        return policy.reshape(policy.shape[0],policy.shape[-1]),value


def get_current_model(net_type="SAGE",hidden_channels=60,hidden_layers=15,policy_layers=2,value_layers=2,in_channels=3,swap_allowed=False,norm=None):
    # return PNA_torch_script(hidden_channels=30,hidden_layers=11,policy_layers=2,value_layers=2,in_channels=3)
    # return PNA_torch_script(hidden_channels=20,hidden_layers=7,policy_layers=2,value_layers=2,in_channels=3)

    if net_type=="SAGE":
        return SAGE_torch_script(hidden_channels=hidden_channels,hidden_layers=hidden_layers,policy_layers=policy_layers,value_layers=value_layers,in_channels=in_channels,swap_allowed=swap_allowed,norm=norm)
    elif net_type=="PNA":
        return PNA_torch_script(hidden_channels=hidden_channels,hidden_layers=hidden_layers,policy_layers=policy_layers,value_layers=value_layers,in_channels=in_channels)
    else:
        raise ValueError("Invalid net type")

