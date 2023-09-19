"""Take the graph-tool graphs from the graph games and convert them to graphs pytorch_geometric understands

Key functions are:
    convert_node_switching_game: Shannon node-switching game -> pytorch_geometric graph
    convert_node_switching_game_back: pytorch_geometric graph -> Shannon node-switching game graph
"""

import torch
from torch_geometric.data import Data
import numpy as np
from graph_tool.all import Graph,VertexPropertyMap
from typing import Tuple, Optional, Union


def graph_to_arrays(graph:Graph) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
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
    
    return node_features,edge_index,targets

def convert_node_switching_game(graph:Graph,target_vp:Optional[VertexPropertyMap]=None,global_input_properties=[],global_output_properties=[],need_backmap=False,old_style=False) -> Data:
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
    if old_style:
        node_features = torch.zeros((n,2+len(global_input_properties)))
        degprop = graph.degree_property_map("total")
        node_features[:,0] = torch.tensor(degprop.fa).float()
        node_features[0,1] = 1
        node_features[1,1] = 1
        for i,p in enumerate(global_input_properties):
            node_features[:,2+i] = p
    else:
        node_features = torch.zeros((n-2,2+len(global_input_properties)))
        neighprop1 = graph.new_vertex_property("int")
        neighprop2 = graph.new_vertex_property("int")
        for neigh in graph.iter_all_neighbors(0):
            neighprop1[neigh] = 1
        for neigh in graph.iter_all_neighbors(1):
            neighprop2[neigh] = 1
        node_features[:,0] = torch.tensor(neighprop1.fa[2:]).float()
        node_features[:,1] = torch.tensor(neighprop2.fa[2:]).float()
        for i,p in enumerate(global_input_properties):
            node_features[:,2+i] = p

    verts = graph.get_vertices()
    assert verts[0] == 0 and verts[1] == 1 # Sanity, first two nodes should be the terminals

    # This is all not super efficient, but as long as I use graph-tool
    # there does not seem to be a better way...
    if old_style:
        vmap = dict(zip(verts,range(0,len(verts))))
    else:
        vmap = dict(zip(verts[2:],range(0,len(verts)-2)))
    edges = graph.get_edges()
    if not old_style:
        edges = [x for x in edges if 0 not in x and 1 not in x]
    if len(edges) == 0:
        edge_index = torch.tensor([[],[]]).long()
    else:
        edge_index = torch.from_numpy(np.vectorize(vmap.get)(edges).astype(int)).transpose(0,1)
        edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),dim=1)  # Make sure, the graph is undirected.

    graph_data = Data(x=node_features,edge_index=edge_index)
    if need_backmap:
        if old_style:
            setattr(graph_data,"backmap",torch.tensor(verts).long())
        else:
            setattr(graph_data,"backmap",torch.tensor(verts[2:]).long())
    if target_vp is not None:
        targray = target_vp.fa
        targets = torch.tensor(targray).unsqueeze(1)
        setattr(graph_data,"y",targets)
    if len(global_output_properties)>0:
        setattr(graph_data,"global_y",torch.tensor(global_output_properties))

    return graph_data

def convert_node_switching_game_back(data:Data) -> Union[Tuple[Graph,VertexPropertyMap],Graph]:
    """Convert a torch_geometric data object into a graph-tool graph for shannon node-switching
    
    Args:
        data: A torch_geometric Data object representing the information from an input graph

    Returns:
        A graph-tool graph for a shannon node-switiching game
        A graph-tool vertex property map for the targets.
    """
    graph = Graph(directed=False)
    graph.gp["m"] = graph.new_graph_property("bool")
    if hasattr(data,"maker_turn"):
        graph.gp["m"] = bool(data.maker_turn)
    else:
        graph.gp["m"] = bool(data.x[0][2])
    graph.add_vertex(len(data.x))
    edge_list = data.edge_index.cpu().numpy().T
    edge_list = np.array([list(x) for x in set([frozenset(x) for x in edge_list.tolist()])])
    graph.add_edge_list(edge_list)
    graph.vp.f = graph.new_vertex_property("bool")
    graph.vp.f.a = True
    if hasattr(data,"y") and data.y is not None:
        tprop = graph.new_vertex_property("double")
        if len(data.y.shape)==1:
            tprop.a = data.y.cpu().numpy()[:]
        else:
            tprop.a = data.y.cpu().numpy()[:,0]
        return graph,tprop
    return graph


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
        A torch_geometric Data object representing the information from an input graph
    """
    node_features,edge_index,targets = graph_to_arrays(graph)

    edge_index = torch.from_numpy(edge_index)
    node_features = torch.from_numpy(node_features)
    targets = torch.from_numpy(targets)

    graph_data = Data(x=node_features,edge_index=edge_index,y=targets)
    return graph_data

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
