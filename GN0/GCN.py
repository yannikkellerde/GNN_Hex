import torch
from torch._C import Graph
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch, Data
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from time import perf_counter

perfs = {
    "start_nn":[],
    "graph_assign":[],
    "propagate":[],
    "final_global":[]
}


class GCNConv_glob(MessagePassing):
    """Bootstrapped from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    to include a global attribute similar to https://arxiv.org/pdf/1806.01261.pdf"""
    def __init__(self, in_channels, out_channels, global_dim):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.node_to_node_lin = torch.nn.Linear(in_channels, out_channels)
        self.glob_to_node_lin = torch.nn.Linear(global_dim, out_channels)
        self.glob_to_glob_lin = torch.nn.Linear(global_dim, global_dim)
        self.node_to_glob_lin = torch.nn.Linear(out_channels, global_dim)

    def forward(self, x, edge_index, global_attr, graph_assignment):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, num_edges]
        # global_attr has shape [num_graphs, global_dim]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.node_to_node_lin(x)

        # Inserted Step: Linearly transform global attributes and add to node features.
        globs = self.glob_to_node_lin(global_attr) 
        for i in range(len(globs)):
            x[graph_assignment == i] += globs[i]
            
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
        graph_parts = torch.stack([torch.max(x[graph_assignment == i],0).values for i in range(len(global_attr))])
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
            graph_assignment = data.batch
            num_graphs = data.num_graphs
        else:
            graph_assignment = torch.zeros(data.num_nodes,dtype=torch.long)
            num_graphs = 1
        x, edge_index  = data.x, data.edge_index

        glob_attr = self.glob_init.repeat(num_graphs,1)

        x, glob_attr = self.convs(x, edge_index, glob_attr, graph_assignment)
        x = F.relu(x)
        for conv in self.conv_between:
            x, glob_attr = conv(x, edge_index, glob_attr, graph_assignment)
            x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x, glob_attr = self.conve(x, edge_index, glob_attr, graph_assignment)

        return torch.sigmoid(x)

