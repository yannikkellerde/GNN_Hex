import torch
from torch_geometric.data import Data
import numpy as np
from graph_tool.all import Graph
from graph_game.graph_tools_game import Graph_game
from typing import Tuple

def convert_graph(graph:Graph) -> Tuple[Data,dict]:
    """Convert a graph-tool graph for a graph_tools_game into torch_geometric data
    
    The torch_geometric data stores the follwing features from the input graphs:
    Graph Features: None
    Node Features: (IsWinsquare, IsOwnedByOnturn, IsOwnedByNotOnturn)
    Edge Features: None
    Targets: (OnturnWinsByForcedMoves,NotOnturnWinsByForcedMoves)

    Args:
        graph: A graph-tool graph for graph_tools_games

    Returns:
        A torch_geometric Data object representing the information from an input graph and
        vertexmap, which is a list with the same length as the number of nodes
        in the GraphTuple that stores the index of the corresponding vertex for
        each graph-tools vertex that represents a square
    """
    node_feat_shape = 3
    node_features = np.zeros((graph.num_vertices(),node_feat_shape),dtype=np.bool)
    edge_index = np.empty((2,graph.num_edges()*2),dtype=np.int64)
    targets = np.empty((graph.num_vertices(),2),dtype=np.bool)
    vertexmap = {}
    vertex_count = 0
    edge_count = 0
    blackturn = graph.gp["b"]
    for ind in graph.iter_vertices():
        node = graph.vertex(ind)
        own_val = graph.vp.o[node]
        vertexmap[vertex_count] = ind
        if own_val != 0:
            node_features[vertex_count][0] = True
        if (own_val == 2 and blackturn) or (own_val == 3 and not blackturn):
            node_features[vertex_count][1] = True
        elif (own_val == 2 and not blackturn) or (own_val == 3 and blackturn):
            node_features[vertex_count][2] = True
        is_won_for_onturn,is_won_for_not_onturn = graph.vp.w[node]
        targets[vertex_count][0] = int(is_won_for_onturn)
        targets[vertex_count][1] = int(is_won_for_not_onturn)
        vertex_count += 1
    rev_vertex_map = {value:key for key,value in vertexmap.items()}

    for edge in graph.get_edges():
        edge_index[0][edge_count] = rev_vertex_map[edge[1]]
        edge_index[1][edge_count] = rev_vertex_map[edge[0]]
        edge_count += 1
        edge_index[0][edge_count] = rev_vertex_map[edge[0]]
        edge_index[1][edge_count] = rev_vertex_map[edge[1]]
        edge_count += 1

    edge_index = torch.from_numpy(edge_index)
    node_features = torch.from_numpy(node_features)
    targets = torch.from_numpy(targets)

    graph_data = Data(x=node_features,edge_index=edge_index,y=targets)
    return graph_data,vertexmap

def convert_graph_back(data:Data) -> Graph_game:
    """Convert a torch_geometric data object into a graph-tool graph for graph_tools_game.
    
    Args:
        data: A torch_geometric Data object representing the information from an input graph

    Returns:
        A graph-tool graph for graph_tools_games
    """
    graph = Graph(directed=False)
    graph.gp["h"] = graph.new_graph_property("long")
    graph.gp["b"] = graph.new_graph_property("bool")
    graph.gp["b"] = True # We will always set black as the onturn player
    owner_prop = graph.new_vertex_property("short")
    graph.vp.o = owner_prop
    filt_prop = graph.new_vertex_property("bool")
    graph.vp.f = filt_prop
    won_prop = graph.new_vertex_property("vector<bool>")
    graph.vp.w = won_prop
    for i,(feat,won) in enumerate(zip(data.x,data.y)):
        vert = graph.add_vertex()
        if feat[0] == 0:
            owner_prop[vert] = 0
        elif feat[1] == 1:
            owner_prop[vert] = 2
        elif feat[2] == 1:
            owner_prop[vert] = 3
        else:
            owner_prop[vert] = 1
        filt_prop[vert] = True
        won_prop[vert] = [won[0],won[1]]
    for i in range(data.edge_index.shape[1]):
        edge = graph.edge(data.edge_index[0][i],data.edge_index[1][i])
        if edge is None:
            graph.add_edge(data.edge_index[0][i],data.edge_index[1][i])
    return graph