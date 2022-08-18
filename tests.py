from graph_game.graph_tools_games import Hex_game
import time
import torch
import numpy as np
from graph_tool.all import adjacency

def test_adjacency():
    def old_method(graph,verts):
        vmap = dict(zip(verts,range(0,len(verts))))
        edges = graph.get_edges()
        if len(edges) == 0:
            edge_index = torch.tensor([[],[]]).long()
        else:
            edge_index = torch.from_numpy(np.vectorize(vmap.get)(edges).astype(int)).transpose(0,1)
            edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),dim=1)
        return edge_index

    def new_method(graph):
        adj = adjacency(graph)
        adj_tensor = torch.from_numpy(adj.indices)
        sources = torch.empty_like(adj_tensor)
        prev_entry = 0
        for i,entry in enumerate(adj.indptr[1:]):
            sources[prev_entry:entry] = i
            prev_entry = entry
        return torch.stack((sources,adj_tensor))

    g = Hex_game(11)
    for i in range(27):
        g.make_move(i+2)
    g.draw_me()

    graph = g.view

    verts = graph.get_vertices()
    
    start = time.perf_counter()
    for i in range(100):
        old_method(graph,verts)
    print(time.perf_counter()-start)
    start = time.perf_counter()
    for i in range(100):
        new_method(graph)
    print(time.perf_counter()-start)


if __name__ == "__main__":
    print(test_adjacency())
