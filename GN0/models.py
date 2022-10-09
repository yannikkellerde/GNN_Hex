import torch
from torch import Tensor
from typing import Optional, List, Tuple, Dict, Type, Union
import torch.nn.functional as F
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch, Data
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.utils
from torch_scatter import scatter, scatter_mean
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, SparseTensor
from torch.nn import Softmax, Sigmoid, Tanh
import numpy as np
from time import perf_counter
from collections import defaultdict
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch.nn import Linear,ModuleList
import copy
from math import sqrt
from argparse import Namespace

perfs = defaultdict(list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FactorizedNoisyLinear(torch.nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        torch.nn.init.uniform_(self.weight_mu, -scale, scale)
        torch.nn.init.uniform_(self.bias_mu, -scale, scale)

        torch.nn.init.constant_(self.weight_sigma, self.sigma_0 * scale)
        torch.nn.init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)


def cachify_gnn(gnn:Type[BasicGNN]):
    class CachifiedGNN(gnn):
        supports_edge_weight = False
        supports_edge_attr = False
        supports_cache = True
        def __init__(self,*args,out_channels: Optional[int] = None,**kwargs):
            super().__init__(*args,out_channels=out_channels,**kwargs)
            self.has_output = out_channels is not None
            self.has_cache = False
            if self.norms is not None and not self.has_output: # Add final norm layer after last hidden layer if not having output
                if len(self.norms)>0:
                    self.norms.append(copy.deepcopy(self.norms[0]))
                else:
                    self.norms.append(copy.deepcopy(kwargs["norm"]))

        def grow_depth(self,additional_layers):
            assert not self.has_output
            self.num_layers+=additional_layers
            for _ in range(additional_layers):
                conv_layer = self.init_conv(self.hidden_channels,self.hidden_channels).to(self.convs[0].lin_l.weight.device)
                conv_layer.lin_l.weight.data[:] = 0
                conv_layer.lin_l.bias.data[:] = 0
                conv_layer.lin_r.weight.data[:] = torch.eye(self.hidden_channels)
                self.convs.append(conv_layer)
            
                if self.norms is not None:
                    new_norm = self.norms[0].__class__(self.hidden_channels).to(self.convs[0].lin_l.weight.device)
                    new_norm.mean_scale.data[:] = 0
                    self.norms.append(new_norm)
            self.has_cache = False

        def grow_width(self,new_width,new_in_channels=None):
            old_convs = self.convs
            if new_in_channels is not None:
                old_in_channels = self.in_channels
                self.in_channels = new_in_channels
            self.convs = ModuleList()
            self.convs.append(self.init_conv(self.in_channels,new_width).to(old_convs[0].lin_l.weight.device))
            for _ in range(self.num_layers-2):
                self.convs.append(self.init_conv(new_width,new_width).to(old_convs[0].lin_l.weight.device))
            if self.has_output:
                self.convs.append(
                    self.init_conv(new_width, self.out_channels).to(old_convs[0].lin_l.weight.device)
                )
            else:
                self.convs.append(
                    self.init_conv(new_width, new_width).to(old_convs[0].lin_l.weight.device))
            for i,(conv,old_conv) in enumerate(zip(self.convs,old_convs)):
                if i == 0:
                    if new_in_channels is None:
                        conv.lin_l.weight.data[:self.hidden_channels,:] = old_conv.lin_l.weight.data
                        conv.lin_r.weight.data[:self.hidden_channels,:] = old_conv.lin_r.weight.data
                    else:
                        conv.lin_l.weight.data[:self.hidden_channels,:old_in_channels] = old_conv.lin_l.weight.data
                        conv.lin_r.weight.data[:self.hidden_channels,:old_in_channels] = old_conv.lin_r.weight.data
                        conv.lin_l.weight.data[:self.hidden_channels,old_in_channels:] = 0
                        conv.lin_r.weight.data[:self.hidden_channels,old_in_channels:] = 0

                else:
                    conv.lin_l.weight.data[:self.hidden_channels,:self.hidden_channels] = old_conv.lin_l.weight.data
                    conv.lin_r.weight.data[:self.hidden_channels,:self.hidden_channels] = old_conv.lin_r.weight.data
                    conv.lin_l.weight.data[:self.hidden_channels,self.hidden_channels:] = 0
                    conv.lin_r.weight.data[:self.hidden_channels,self.hidden_channels:] = 0
                conv.lin_l.bias.data[:self.hidden_channels] = old_conv.lin_l.bias.data


            if self.norms is not None:
                new_norms = ModuleList()
                for i in range(self.num_layers-1 if self.has_output else self.num_layers):
                    new_norm = self.norms[i].__class__(new_width).to(old_convs[0].lin_l.weight.device)
                    new_norm.weight.data[:self.hidden_channels] = self.norms[i].weight.data
                    new_norm.bias.data[:self.hidden_channels] = self.norms[i].bias.data
                    new_norm.mean_scale.data[:self.hidden_channels] = self.norms[i].mean_scale.data
                    new_norms.append(new_norm)
                self.norms = new_norms

            self.has_cache = False
            self.hidden_channels = new_width


        def export_norm_cache(self) -> Tuple[Tensor,Tensor]:
            if self.norms is None:
                return
            assert self.has_cache
            mean_caches = []
            var_caches = []
            for norm in self.norms:
                mean_caches.append(norm.mean_cache)
                var_caches.append(norm.var_cache)
            return torch.stack(mean_caches),torch.stack(var_caches)

        def import_norm_cache(self,mean_cache,var_cache):
            if self.norms is None:
                return
            self.has_cache = True
            for i,norm in enumerate(self.norms):
                norm.mean_cache = mean_cache[i].to(norm.weight.device)
                norm.var_cache = var_cache[i].to(norm.weight.device)

        def forward(self,x: Tensor,edge_index: Adj,*,edge_weight: OptTensor = None,edge_attr: OptTensor = None,set_cache: bool = False) -> Tensor:
            if set_cache:
                self.has_cache = True
            xs: List[Tensor] = []
            for i in range(self.num_layers):
                # Tracing the module is not allowed with *args and **kwargs 
                # As such, we rely on a static solution to pass optional edge
                # weights and edge attributes to the module.
                if self.supports_edge_weight and self.supports_edge_attr:
                    x = self.convs[i](x, edge_index, edge_weight=edge_weight,edge_attr=edge_attr)
                elif self.supports_edge_weight:
                    x = self.convs[i](x, edge_index, edge_weight=edge_weight)
                elif self.supports_edge_attr:
                    x = self.convs[i](x, edge_index, edge_attr=edge_attr)
                else:
                    x = self.convs[i](x, edge_index)
                if i == self.num_layers - 1 and self.jk_mode is None and self.has_output:
                    break
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.norms is not None:
                    x = self.norms[i](x,set_cache=set_cache,use_cache=self.has_cache and not self.training)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if hasattr(self, 'jk'):
                    xs.append(x)

            x = self.jk(xs) if hasattr(self, 'jk') else x
            x = self.lin(x) if hasattr(self, 'lin') else x
            return x
    return CachifiedGNN

class ActionValue(torch.nn.Module):
    def __init__(self,GNN:Type[BasicGNN],**gnn_kwargs):
        super().__init__()
        gnn_kwargs["out_channels"] = 1
        self.gnn = GNN(**gnn_kwargs)
        self.supports_cache = hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache
        self.value_activation = torch.nn.Tanh()

    def export_norm_cache(self,*args,**kwargs):
        return self.gnn.export_norm_cache(*args,**kwargs)
    
    def import_norm_cache(self,*args,**kwargs):
        return self.gnn.import_norm_cache(*args,**kwargs)

    def forward(self,x:Tensor,edge_index:Adj,set_cache:bool=False) -> Tensor:
        if hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache:
            res = self.gnn(x,edge_index,set_cache=set_cache)
        else:
            res = self.gnn(x,edge_index)
        return self.value_activation(res)

class HeadNetwork(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,GNN:Type[BasicGNN],noisy_dqn=True,noise_sigma=0,**gnn_kwargs):
        super().__init__()
        self.gnn = GNN(in_channels=in_channels,hidden_channels=hidden_channels,**gnn_kwargs)
        self.supports_cache = hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache
        self.value_head = Linear(hidden_channels,1)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        if noisy_dqn:
            self.linear = FactorizedNoisyLinear(hidden_channels,out_channels,noise_sigma)
        else:
            self.linear = torch.nn.Linear(hidden_channels,out_channels)

    def grow_width(self,new_width,new_in_channels=None):
        assert isinstance(self.linear,torch.nn.Linear)
        self.gnn.grow_width(new_width,new_in_channels=new_in_channels)

        old_linear = self.linear
        self.linear = torch.nn.Linear(new_width,self.out_channels).to(old_linear.weight.device)
        self.linear.weight.data.fill_(0)
        self.linear.weight.data[:,:self.hidden_channels] = old_linear.weight.data
        self.linear.bias.data[:] = old_linear.bias.data[:]

        old_head = self.value_head
        self.value_head = torch.nn.Linear(new_width,self.out_channels).to(old_head.weight.device)
        self.value_head.weight.data.fill_(0)
        self.value_head.weight.data[:,:self.hidden_channels] = old_head.weight.data
        self.value_head.bias.data[:] = old_head.bias.data[:]

        self.hidden_channels = new_width

    def grow_depth(self,additional_layers):
        self.gnn.grow_depth(additional_layers)

    def export_norm_cache(self,*args,**kwargs):
        return self.gnn.export_norm_cache(*args,**kwargs)
    
    def import_norm_cache(self,*args,**kwargs):
        return self.gnn.import_norm_cache(*args,**kwargs)

    def forward(self,x:Tensor,edge_index:Tensor,graph_indices,advantages_only=False,set_cache=False):
        if hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache:
            x = self.gnn(x,edge_index,set_cache=set_cache)
        else:
            x = self.gnn(x,edge_index)
        
        advantages = self.linear(x)
        if advantages_only:
            return advantages

        graph_parts = scatter(x,graph_indices,dim=0,reduce="sum")
        value = self.value_head(graph_parts)
        return advantages,value

class DuellingTwoHeaded(torch.nn.Module):
    def __init__(self,GNN:Type[BasicGNN],advantage_head,gnn_kwargs,head_kwargs):
        super().__init__()
        self.gnn = GNN(**gnn_kwargs)
        self.supports_cache = hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache
        self.value_activation = Tanh()
        self.advantage_activation = Tanh()
        self.maker_head = advantage_head(in_channels=gnn_kwargs["hidden_channels"],hidden_channels=gnn_kwargs["hidden_channels"],out_channels=1,**head_kwargs)
        self.breaker_head = advantage_head(in_channels=gnn_kwargs["hidden_channels"],hidden_channels=gnn_kwargs["hidden_channels"],out_channels=1,**head_kwargs)

    def grow_depth(self,additional_layers):
        self.gnn.grow_depth(additional_layers)

    def grow_width(self,new_width):
        self.gnn.grow_width(new_width)
        self.maker_head.grow_width(new_width,new_in_channels=new_width)
        self.breaker_head.grow_width(new_width,new_in_channels=new_width)

    def export_norm_cache(self,*args):
        cache_list = []
        if hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache:
            cache_list.append(self.gnn.export_norm_cache(*args))
        if hasattr(self.maker_head,"supports_cache") and self.maker_head.supports_cache:
            cache_list.append(self.maker_head.export_norm_cache(*args))
        if hasattr(self.breaker_head,"supports_cache") and self.breaker_head.supports_cache:
            cache_list.append(self.breaker_head.export_norm_cache(*args))
        return cache_list
    
    def import_norm_cache(self,*args):
        ind = 0
        if hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache:
            if args[ind] is not None:
                self.gnn.import_norm_cache(*args[ind])
            ind+=1
        if hasattr(self.maker_head,"supports_cache") and self.maker_head.supports_cache:
            if args[ind] is not None:
                self.maker_head.import_norm_cache(*args[ind])
            ind+=1
        if hasattr(self.breaker_head,"supports_cache") and self.breaker_head.supports_cache:
            if args[ind] is not None:
                self.breaker_head.import_norm_cache(*args[ind])

    def forward(self,x:Tensor,edge_index:Adj,graph_indices:Optional[Tensor]=None,ptr:Optional[Tensor]=None,set_cache:bool=False,advantages_only=False,seperate=False) -> Union[Tensor,Tuple[Tensor,Tensor]]:
        assert torch.all(x[:,2] == x[0,2])
        is_maker = x[0,2]
        x = x[:,:2]

        if graph_indices is None:
            graph_indices = x.new_zeros(x.size(0), dtype=torch.long)    
        
        if hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache:
            embeds = self.gnn(x,edge_index,set_cache=set_cache)
        else:
            embeds = self.gnn(x,edge_index)

        if is_maker==1:
            if hasattr(self.maker_head,"supports_cache") and self.maker_head.supports_cache:
                head_res = self.maker_head(embeds,edge_index,graph_indices,advantages_only=advantages_only,set_cache=set_cache)
            else:
                head_res = self.maker_head(embeds,edge_index,graph_indices,advantages_only=advantages_only)
        else:
            if hasattr(self.breaker_head,"supports_cache") and self.breaker_head.supports_cache:
                head_res = self.breaker_head(embeds,edge_index,graph_indices,advantages_only=advantages_only,set_cache=set_cache)
            else:
                head_res = self.breaker_head(embeds,edge_index,graph_indices,advantages_only=advantages_only)

        if advantages_only:
            advantages = 2*self.advantage_activation(head_res) # Advantage range: -2, 2
            return advantages
        else:
            advantages = 2*self.advantage_activation(head_res[0]) # Advantage range: -2, 2
            value = head_res[1]
            
        value = self.value_activation(value) # Value range: -1, 1

        batch_size = int(graph_indices.max()) + 1

        adv_means = scatter(advantages,graph_indices,dim=0,dim_size=batch_size,reduce="mean")
        
        if seperate:
            return value.squeeze(),(advantages - adv_means.index_select(0, graph_indices)).squeeze()
        else:
            # No final activation -> Outputs range from -5 to +5
            return (value.index_select(0,graph_indices) + (advantages - adv_means.index_select(0, graph_indices))).squeeze()
    
    def simple_forward(self,data:Union[Data,Batch]):
        if isinstance(data,Batch):
            return self.forward(data.x,data.edge_index,data.batch,data.ptr)
        else:
            return self.forward(data.x,data.edge_index)

class Duelling():
    def __init__(self,GNN,*args,advantage_head=None,**kwargs):
        super().__init__()
        if advantage_head is not None:
            kwargs["policy_head"] = advantage_head
        self.gnn = GNN(**kwargs)
        self.supports_cache = hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache
        self.advantage_head = kwargs["policy_head"](kwargs["hidden_channels"],1,)
        self.value_head = Linear(kwargs["hidden_channels"],1)
        self.value_activation = Tanh()
        self.advantage_activation = Tanh()
        self.final_activation = Tanh()

    def forward(self,x:Tensor,edge_index:Adj,graph_indices:Optional[Tensor]=None,ptr:Optional[Tensor]=None,set_cache:bool=False,advantages_only=False) -> Union[Tensor,Tuple[Tensor,Tensor]]:
        if hasattr(self.gnn,"supports_cache") and self.gnn.supports_cache:
            embeds = self.gnn(x,edge_index,set_cache=set_cache)
        else:
            embeds = self.gnn(x,edge_index)

        advantages = self.advantage_head(embeds,edge_index)
        advantages = 2*self.advantage_activation(advantages) # Advantage range: -2, 2


        # color_multiplier = (x[:,2]*2-1).unsqueeze(1)  # This ensures that value is computed in terms of the player that is on turn. Hacky, but whatever.
        # advantages = advantages*color_multiplier
        
        if advantages_only:
            return advantages

        if graph_indices is None:
            graph_indices = x.new_zeros(x.size(0), dtype=torch.long)    
        
        graph_parts = scatter(embeds,graph_indices,dim=0,reduce="sum")
        value = self.value_head(graph_parts)
        value = self.value_activation(value) # Value range: -1, 1

        batch_size = int(graph_indices.max()) + 1

        # value_color_multiplier = (x[:,2:3][ptr[:-1]]*2-1)
        # value = value*value_color_multiplier

        adv_means = scatter(advantages,graph_indices,dim=0,dim_size=batch_size,reduce="mean")
        
        return self.final_activation((value.index_select(0,graph_indices) + (advantages - adv_means.index_select(0, graph_indices))).squeeze())
    
    def simple_forward(self,data:Union[Data,Batch]):
        if isinstance(data,Batch):
            return self.forward(data.x,data.edge_index,data.batch,data.ptr)
        else:
            return self.forward(data.x,data.edge_index)


class CachedGraphNorm(GraphNorm):
    supports_cache=True
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__(in_channels,eps)
        self.mean_cache = None
        self.var_cache = None

    def forward(self, x: Tensor, batch: Optional[Tensor] = None, set_cache=False, use_cache=False) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max()) + 1
        if use_cache and not set_cache:
            mean = self.mean_cache
        else:
            mean = scatter_mean(x, batch, dim=0, dim_size=batch_size)
            if set_cache:
                self.mean_cache = mean
        out = x - mean.index_select(0, batch) * self.mean_scale
        if use_cache and not set_cache:
            var = self.var_cache
        else:
            var = scatter_mean(out.pow(2), batch, dim=0, dim_size=batch_size)
            if set_cache:
                self.var_cache = var
        std = (var + self.eps).sqrt().index_select(0, batch)
        return self.weight * out / std + self.bias

class GCNConv_glob(MessagePassing):
    """Bootstrapped from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    to include a global attribute similar to https://arxiv.org/pdf/1806.01261.pdf"""
    def __init__(self, in_channels, out_channels, global_dim):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.node_to_node_lin = torch.nn.Linear(in_channels, out_channels)
        self.glob_to_node_lin = torch.nn.Linear(global_dim, out_channels)
        self.glob_to_glob_lin = torch.nn.Linear(global_dim, global_dim)
        self.node_to_glob_lin = torch.nn.Linear(out_channels, global_dim)

    def forward(self, x, edge_index, global_attr, binary_matrix, graph_indices):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, num_edges]
        # global_attr has shape [num_graphs, global_dim]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.node_to_node_lin(x)
        
        # Inserted Step: Linearly transform global attributes and add to node features.
        globs = self.glob_to_node_lin(global_attr) 

        # Slow Aggregation:
        # for i in range(len(globs)):
        #     x[graph_slices[i]:graph_slices[i+1]] += globs[i]

        # Fast parallel aggregation:
        x += torch.matmul(binary_matrix,globs)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4-5: Start propagating messages.
        x = self.propagate(edge_index, x=x, norm=norm)

        # Inserted Step: Update global attribute
        global_attr = self.glob_to_glob_lin(global_attr)

        # Slow aggregation:
        # graph_parts = torch.stack([torch.max(x[graph_slices[i]:graph_slices[i+1]],0).values for i in range(len(global_attr))])
        
        # Quick cuda aggregation:
        graph_parts = scatter(x, graph_indices, dim=0, reduce="max")
        
        # Inserted step: Update global attribute using node features (symmetrically reduced)
        global_attr = global_attr + self.node_to_glob_lin(graph_parts)

        return x,global_attr

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCN(torch.nn.Module):
    def __init__(self,num_node_features,label_dimension,conv_layers=2,conv_dim=16,output_activation=lambda x:x,parameter_sharing=False,instance_norm=False):
        super().__init__()
        self.convs = GCNConv(num_node_features, conv_dim)
        conv_between = []
        if parameter_sharing:
            gcn_conv = GCNConv(conv_dim,conv_dim,aggr="add")
            conv_between = [gcn_conv for _ in range(conv_layers-2)]
        else:
            for _ in range(conv_layers-2):
                conv_between.append(GCNConv(conv_dim, conv_dim,aggr="add"))
        self.instance_norm = instance_norm
        if self.instance_norm:
            i_norms = []
            for _ in range(conv_layers-1):
                i_norms.append(torch.nn.InstanceNorm1d(conv_dim))
            self.i_norms = torch.nn.ModuleList(i_norms)
        self.conv_between = torch.nn.ModuleList(conv_between)
        self.conve = GCNConv(conv_dim, label_dimension)
        self.output_activation = output_activation

    def forward(self, x, edge_index):

        x = self.convs(x, edge_index)
        x = F.relu(x)
        if self.instance_norm:
            x = self.i_norms[0](x)
        for i,conv in enumerate(self.conv_between):
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.instance_norm:
                x = self.i_norms[i+1](x)
        #x = F.dropout(x, training=self.training)
        x = self.conve(x, edge_index)
        return self.output_activation(x)

class GCN_with_glob(torch.nn.Module):
    def __init__(self,num_node_features,label_dimension,conv_layers=2,conv_dim=16,global_dim=16):
        super().__init__()
        self.convs = GCNConv_glob(num_node_features, conv_dim, global_dim)
        conv_between = []
        for _ in range(conv_layers-2):
            conv_between.append(GCNConv_glob(conv_dim, conv_dim, global_dim))
        self.conv_between = torch.nn.ModuleList(conv_between)
        self.conve = GCNConv_glob(conv_dim, label_dimension, global_dim)
        self.glob_init = torch.nn.Parameter(torch.randn(1,global_dim))

    def forward(self, data:Data):
        if isinstance(data,Batch):
            num_graphs = data.num_graphs
            binary_matrix = torch.zeros(data.x.size(0),num_graphs).to(device)
            for i in range(len(data.ptr)-1):
                binary_matrix[data.ptr[i]:data.ptr[i+1],i] = 1
            graph_indices = data.batch
        else:
            binary_matrix = torch.ones(data.x.size(0),1).to(device)
            num_graphs = 1
            graph_indices = torch.zeros(data.x.size(0)).long().to(device)
        x, edge_index  = data.x, data.edge_index

        glob_attr = self.glob_init.repeat(num_graphs,1)

        x, glob_attr = self.convs(x, edge_index, glob_attr, binary_matrix, graph_indices)
        x = F.relu(x)
        for conv in self.conv_between:
            x, glob_attr = conv(x, edge_index, glob_attr, binary_matrix, graph_indices)
            x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x, glob_attr = self.conve(x, edge_index, glob_attr, binary_matrix, graph_indices)
        return torch.sigmoid(x)


class PolicyValue(torch.nn.Module):
    supports_cache = False
    def __init__(self,GNN,hidden_channels,hidden_layers,policy_layers,value_layers,in_channels=3,**gnn_kwargs):
        super().__init__()
        self.gnn = GNN(in_channels=in_channels,hidden_channels=hidden_channels,num_layers=hidden_layers,**gnn_kwargs)

        self.maker_modules = torch.nn.ModuleDict()
        self.breaker_modules = torch.nn.ModuleDict()

        self.maker_modules["value_head"] = GNN(in_channels=hidden_channels,hidden_channels=hidden_channels,num_layers=value_layers)
        self.breaker_modules["value_head"] = GNN(in_channels=hidden_channels,hidden_channels=hidden_channels,num_layers=value_layers)
        self.maker_modules["policy_head"] = GNN(in_channels=hidden_channels,hidden_channels=hidden_channels,num_layers=policy_layers)
        self.breaker_modules["policy_head"] = GNN(in_channels=hidden_channels,hidden_channels=hidden_channels,num_layers=policy_layers)
        self.maker_modules["linear"] = torch.nn.Linear(hidden_channels,1)
        self.breaker_modules["linear"] = torch.nn.Linear(hidden_channels,1)

        self.value_activation = torch.nn.Sigmoid()

    def grow(self,extra_body_layers=0,extra_width=0,extra_value_head_layers=0,extra_policy_head_layers=0):
        if extra_width>0:
            new_width = self.gnn.hidden_channels+extra_width
            self.gnn.grow_width(new_width=new_width)
            self.maker_modules["value_head"].grow_width(
                    new_width=new_width,
                    new_in_channels=new_width
            )
            self.maker_modules["policy_head"].grow_width(
                    new_width=new_width,
                    new_in_channels=new_width
            )
            old_linear = self.maker_modules["linear"]
            self.maker_modules["linear"] = torch.nn.Linear(
                    new_width,
                    1
            ).to(old_linear.weight.device)
            self.maker_modules["linear"].weight.data.fill_(0)
            self.maker_modules["linear"].weight.data[:,:self.hidden_channels] = old_linear.weight.data
            self.maker_modules["linear"].bias.data[:] = old_linear.bias.data[:]
            

            old_linear = self.breaker_modules["linear"]
            self.breaker_modules["linear"] = torch.nn.Linear(new_width,1).to(old_linear.weight.device)
            self.breaker_modules["linear"].weight.data.fill_(0)
            self.breaker_modules["linear"].weight.data[:,:self.hidden_channels] = old_linear.weight.data
            self.breaker_modules["linear"].bias.data[:] = old_linear.bias.data[:]
        if extra_body_layers>0:
            self.gnn.grow_depth(extra_body_layers)
        if extra_policy_head_layers>0:
            self.maker_modules["policy_head"].grow_depth(extra_policy_head_layers)
            self.breaker_modules["policy_head"].grow_depth(extra_policy_head_layers)
        if extra_value_head_layers>0:
            self.maker_modules["value_head"].grow_depth(extra_policy_head_layers)
            self.breaker_modules["value_head"].grow_depth(extra_policy_head_layers)



    def forward(self,data):
        x = data.x
        assert torch.all(x[:,2] == x[0,2])
        if x[0,2]==1:
            modules = self.maker_modules
        else:
            modules = self.breaker_modules
        x = x[:,2]

        if hasattr(data,"batch") and data.batch is not None:
            graph_indices = data.batch
        else:
            graph_indices = x.new_zeros(x.size(0), dtype=torch.long)    

        embeds = self.gnn(x,data.edge_index)

        pi = modules["policy"](embeds,data.edge_index)
        value_embeds = modules["value"](embeds,data.edge_index)

        pi = torch.log(torch_geometric.utils.softmax(pi,index=graph_indices))
        graph_parts = scatter(value_embeds,graph_indices,dim=0,reduce="sum")

        value = modules["linear"](graph_parts)
        value = self.value_activation(value)
        return pi.squeeze(),value.squeeze()


def get_pre_defined(name,args=None) -> torch.nn.Module:
    if name == "sage+norm":
        body_model = cachify_gnn(GraphSAGE) 
        model = Duelling(body_model,in_channels=3,num_layers=13,hidden_channels=32,norm=CachedGraphNorm(32),act="relu",advantage_head=SAGEConv)
    elif name == "sage":
        body_model = cachify_gnn(GraphSAGE) 
        model = Duelling(body_model,in_channels=3,num_layers=13,hidden_channels=32,norm=None,act="relu",advantage_head=SAGEConv)
    elif name == "action_value":
        model = ActionValue(cachify_gnn(GraphSAGE),in_channels=3,out_channels=1,num_layers=13,hidden_channels=32,norm=CachedGraphNorm(32),act="relu")
    elif name == "two_headed":
        model = DuellingTwoHeaded(cachify_gnn(GraphSAGE),HeadNetwork,
            gnn_kwargs=dict(
                in_channels=2,
                num_layers=args.num_layers,
                hidden_channels=args.hidden_channels,
                norm=CachedGraphNorm(args.hidden_channels) if args.norm else None,
                act="relu"
            ),head_kwargs=dict(
                GNN=cachify_gnn(GraphSAGE),
                num_layers=args.num_head_layers if hasattr(args,"num_head_layers") else 2,
                noisy_dqn=args.noisy_dqn,
                noise_sigma=args.noisy_sigma0,
                norm=CachedGraphNorm(args.hidden_channels) if args.norm else None,
                act="relu"
            ))
    elif name == "policy_value_small":
        model = PolicyValue(cachify_gnn(GraphSAGE),hidden_channels=25,hidden_layers=8,policy_layers=2,value_layers=2,norm=None,act="relu")
    else:
        print(name)
        raise NotImplementedError
    return model

if __name__ == "__main__":
    get_pre_defined("two_headed",args=Namespace(num_layers=3,hidden_channels=8,norm=True,noisy_dqn=False,noisy_sigma0=False))
