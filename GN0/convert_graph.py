import torch
from torch_geometric.data import Data
import numpy as np
from graph_tool.all import Graph,VertexPropertyMap
from graph_game.winpattern_game import Winpattern_game
from typing import Tuple

def graph_to_arrays(graph:Graph) -> Tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    """ Convert a graph-tool graph into node_features, edge_index, targets, and vertexmap
    Args:
        graph: A graph-tool graph for graph_tools_games

    Returns:
        node_features: A boolean numpy array of shape (num_vertices,3) representing the node features
        edge_index: A numpy array of shape (2,num_edges*2) representing the edge indices
        targets: A boolean numpy array of shape (num_vertices,2) representing the targets
        vertexmap: A dictionary mapping the torch_geometric data vertex indices to the graph-tool graph vertex indices
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
    
    return node_features,edge_index,targets,vertexmap

def convert_node_switching_game(graph:Graph,target_vp:VertexPropertyMap):
    """Convert a graph-tool graph for a shannon node switching game into torch_geometric data

    The torch_geometric data stores the follwing features from the input graphs:
    Graph Features: who's turn is it (0 for breaker, 1 for maker)
    Node Features: 1 if terminal otherwise 0
    Edge Features: None
    Targets: Given by target_vp

    Args:
        graph: A graph-tool graph for a shannon node-switching game.
               Terminal nodes are assumed to be in positions 0 and 1.
        target_vp: A vertex property map to use as node value targets
    
    Returns:
        A torch_geometric Data object representing the graph
    """
    n = graph.num_vertices()
    node_features = torch.zeros(n)
    node_features[0] = 1
    node_features[1] = 1
    verts = graph.get_vertices()

    # This is all not super efficient, but as long as I use graph-tool
    # there does not seem to be a better way...
    vmap = dict(zip(verts,range(0,len(verts))))
    edges = graph.get_edges()
    edge_index = torch.from_numpy(np.vectorize(vmap.get)(edges))
    targray = target_vp.a
    print(targray.shape,target_vp.get_array()[:].shape,verts.shape)
    targets = torch.tensor([targray[i] for i in verts])

    graph_data = Data(x=node_features,edge_index=edge_index,y=targets,maker_turn=int(graph.gp["m"]))
    return graph_data

def convert_node_switching_game_back(data:Data) -> Tuple[Graph,VertexPropertyMap]:
    """Convert a torch_geometric data object into a graph-tool graph for shannon node-switching
    
    Args:
        data: A torch_geometric Data object representing the information from an input graph

    Returns:
        A graph-tool graph for a shannon node-switiching game
        A graph-tool vertex property map for the targets.
    """
    graph = Graph(directed=False)
    graph.gp["m"] = graph.new_graph_property("bool")
    graph.gp["m"] = bool(data.maker_turn)
    graph.add_vertex(len(data.x))
    graph.add_edge_list(data.edge_index)
    tprop = graph.new_vertex_property("double")
    tprop.a = data.y
    return graph,tprop


def convert_winpattern_game(graph:Graph) -> Tuple[Data,dict]:
    """Convert a graph-tool graph for a winpattern_game into torch_geometric data
    
    The torch_geometric data stores the follwing features from the input graphs:
    Graph Features: None
    Node Features: (IsWinsquare, IsOwnedByOnturn, IsOwnedByNotOnturn)
    Edge Features: None
    Targets: (OnturnWinsByForcedMoves,NotOnturnWinsByForcedMoves)

    Args:
        graph: A graph-tool graph for a winpattern game 

    Returns:
        A torch_geometric Data object representing the information from an input graph and
        vertexmap, which maps the torch_geometric data vertex indices to the graph-tool graph vertex indices
    """
    node_features,edge_index,targets,vertexmap = graph_to_arrays(graph)

    edge_index = torch.from_numpy(edge_index)
    node_features = torch.from_numpy(node_features)
    targets = torch.from_numpy(targets)

    graph_data = Data(x=node_features,edge_index=edge_index,y=targets)
    return graph_data,vertexmap

def convert_winpattern_game_back(data:Data) -> Graph:
    """Convert a torch_geometric data object into a graph-tool graph for winpattern_game 
    
    Args:
        data: A torch_geometric Data object representing the information from an input graph

    Returns:
        A graph-tool graph for a winpattern game
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
