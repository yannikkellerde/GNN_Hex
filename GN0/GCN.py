import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self,num_node_features,label_dimension,conv_layers=2,conv_dim=16):
        super().__init__()
        self.convs = GCNConv(num_node_features, conv_dim)
        conv_between = []
        for _ in range(conv_layers-2):
            conv_between.append(GCNConv(conv_dim, conv_dim))
        self.conv_between = torch.nn.ModuleList(conv_between)
        self.conve = GCNConv(conv_dim, label_dimension)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.convs(x, edge_index)
        x = F.relu(x)
        for conv in self.conv_between:
            x = conv(x, edge_index)
            x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conve(x, edge_index)

        return torch.sigmoid(x)