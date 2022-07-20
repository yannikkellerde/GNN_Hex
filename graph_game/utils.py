from graph_tool.all import Graph, Vertex
from typing import List

def fully_connect_lists(g:Graph,l1:List[Vertex],l2:List[Vertex]):
    for v1 in l1:
        for v2 in l2:
            g.add_edge(v1,v2)
