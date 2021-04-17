import torch
from torch_geometric import Data
import numpy as np
from graph_tool.all import Graph


def convert_graph(graph:Graph):
    """Convert a graph-tool graph for a graph_tools_game into torch_geometric data
    
    The torch_geometric data stores the follwing features from the input graphs:
    Graph Features: None
    Node Features: (IsWinsquare, IsOwnedByOnturn, IsOwnedByNotOnturn)
    Edge Features: None

    Args:
        graphs: An iterable of graph-tool graphs for graph_tools_games

    Returns:
        A torch_geometric Data object representing the information from an input graph and
        vertexmap, which is a list with the same length as the number of nodes
        in the GraphTuple that stores the index of the corresponding vertex for
        each graph-tools vertex that represents a square
    """
    node_feat_shape = 3
    node_features = np.zeros((graph.num_vertices(),node_feat_shape))
    edge_index = np.empty((graph.num_edges()*2,2))
    vertexmap = -np.ones(graph.num_vertices())
    vertex_count = 0
    edge_count = 0
    blackturn = graph.gp["b"]
    for ind in graph.iter_vertices():
        node = graph.vertex(ind)
        own_val = graph.vp.o[node]
        if own_val == 0:
            vertexmap[vertex_count] = ind
        else:
            node_features[vertex_count][0] = 1
        if (own_val == 2 and blackturn) or (own_val == 3 and not blackturn):
            node_features[vertex_count][1] = 1
        elif (own_val == 2 and not blackturn) or (own_val == 3 and blackturn):
            node_features[vertex_count][2] = 1
        vertex_count += 1
    for edge in graph.get_edges():
        edge_index[edge_count][0] = edge[1]
        edge_index[edge_count][1] = edge[0]
        edge_count += 1
        edge_index[edge_count][0] = edge[0]
        edge_index[edge_count][1] = edge[1]
        edge_count += 1


    edge_index = torch.from_numpy(edge_index,dtype=torch.long)
    node_features = torch.from_numpy(node_features,dtype=torch.float)

    graph_data = Data(x=node_features,edge_index=edge_index)
    return graph_data,vertexmap