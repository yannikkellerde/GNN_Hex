import torch
from torch._C import Graph
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch, Data
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
import numpy as np
from time import perf_counter
from collections import defaultdict

perfs = defaultdict(list)


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

        
        start = perf_counter()

        # Step 4-5: Start propagating messages.
        x = self.propagate(edge_index, x=x, norm=norm)

        perfs["propagate"].append(perf_counter() - start)

        # Inserted Step: Update global attribute
        global_attr = self.glob_to_glob_lin(global_attr)

        # Slow aggregation:
        # graph_parts = torch.stack([torch.max(x[graph_slices[i]:graph_slices[i+1]],0).values for i in range(len(global_attr))])
        
        # Quick cuda aggregation:
        graph_parts = scatter(x, graph_indices, dim=0, reduce="max")
        
        
        global_attr = global_attr + self.node_to_glob_lin(graph_parts)

        return x,global_attr

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCN(torch.nn.Module):
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
            binary_matrix = torch.zeros(data.x.size(0),num_graphs).cuda()
            for i in range(len(data.ptr)-1):
                binary_matrix[data.ptr[i]:data.ptr[i+1],i] = 1
            graph_indices = data.batch
        else:
            binary_matrix = torch.ones(data.x.size(0),1).cuda()
            num_graphs = 1
            graph_indices = torch.zeros(data.x.size(0)).long().cuda()
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

