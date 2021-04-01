"""Convert a graph-tool graphs into a graph network GraphTuple
Graph Features: None
Node Features: IsWinsquare, IsOwnedByOnturn, IsOwnedByNotOnturn => 3D
Edge Features: None
"""
import graph_nets as gn
import numpy as np

def convert(graphs):
    node_feat_shape = 3
    n_nodes = np.array([g.num_vertices() for g in graphs])
    node_features = np.zeros((np.sum(n_nodes,node_feat_shape)))
    n_edges = np.array([g.num_edges() for g in graphs])
    edge_features = None
    receivers = np.empty((np.sum(n_edges),),dtype=np.int32)
    senders = np.empty_like(receivers)
    global_features = None
    vertex_count = 0
    edge_count = 0
    for i,graph in enumerate(graphs):
        blackturn = graph.gp["b"]
        for node in graph.vertices():
            own_val = graph.vp.o[node]
            if own_val != 0:
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

    GN = gn.GraphTuple(nodes=node_features,edges=None,globals=None,receivers=receivers,
                       senders=senders,n_edges=n_edges,n_nodes=n_nodes)
    return GN