"""Convert graph-tool graphs into a graph network GraphTuple
Graph Features: Zero
Node Features: IsWinsquare, IsOwnedByOnturn, IsOwnedByNotOnturn
Edge Features: Zero
"""
import graph_nets as gn
import numpy as np
from typing import Iterable
from graph_tool.all import Graph
import tensorflow as tf

def convert_graph(graphs:Iterable[Graph]):
    """Convert graph-tool graphs for graph_tools_games into a graph network GraphTuple
    
    The GraphTuple stores the following Graph, Node and Edge features extracted
    from the input graphs:

    Graph Features: None
    Node Features: (IsWinsquare, IsOwnedByOnturn, IsOwnedByNotOnturn)
    Edge Features: None

    Args:
        graphs: An iterable of graph-tool graphs for graph_tools_games

    Returns:
        A GraphTuple representing the information from all input graphs and
        vertexmap, which is a list with the same lenght as the number of nodes
        in the GraphTuple that stores the index of the corresponding vertex for
        each graph-tools vertex that represents a square
    """
    node_feat_shape = 3
    n_nodes = np.array([g.num_vertices() for g in graphs])
    node_features = np.zeros((np.sum(n_nodes),node_feat_shape))
    n_edges = np.array([g.num_edges()*2 for g in graphs])
    receivers = np.empty((np.sum(n_edges),),dtype=np.int32)
    senders = np.empty_like(receivers)
    vertexmap = -np.ones(np.sum(n_nodes))
    vertex_count = 0
    edge_count = 0
    for graph in graphs:
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
            receivers[edge_count] = edge[1]
            senders[edge_count] = edge[0]
            edge_count += 1
            receivers[edge_count] = edge[0]
            senders[edge_count] = edge[1]
            edge_count += 1

    node_features = tf.convert_to_tensor(node_features,dtype=tf.float32)
    receivers = tf.convert_to_tensor(receivers,dtype=tf.float32)
    senders = tf.convert_to_tensor(senders,dtype=tf.float32)
    n_edges = tf.convert_to_tensor(n_edges,dtype=tf.float32)
    n_nodes = tf.convert_to_tensor(n_nodes,dtype=tf.float32)
    GN = gn.graphs.GraphsTuple(nodes=node_features,edges=None,globals=None,receivers=receivers,
                       senders=senders,n_edge=n_edges,n_node=n_nodes)
    GN = gn.utils_tf.set_zero_edge_features(GN,1,dtype=tf.float32)
    GN = gn.utils_tf.set_zero_global_features(GN,1,dtype=tf.float32)
    return GN,vertexmap