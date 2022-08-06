from collections import Counter
from hashlib import blake2b
from graph_tool.all import VertexPropertyMap,GraphPropertyMap,Graph
from time import perf_counter
import numpy as np
from typing import Union

def wl_hash(G:Graph, node_property:VertexPropertyMap, graph_property:Union[None,GraphPropertyMap]=None, iterations=3, digest_size=7) -> int:
    ind_map = [int(x) for x in G.vertices()]
    rev_ind = {key:value for value,key in enumerate(ind_map)}
    def nei_agg(G, n, node_labels):
        x = [node_labels[n]]
        for nei in G.get_all_neighbors(ind_map[n]):
            x.append(node_labels[rev_ind[nei]])
        return ''.join(sorted(x))

    def wl_step(G, labels):
        """
            Aggregate neighbor labels and edge label.
        """
        new_labels = []
        for n in range(G.num_vertices()):
            new_labels.append(nei_agg(G, n, labels))
        return new_labels

    items = []
    node_labels = [str(x) for x in node_property.get_array()[ind_map]]

    for k in range(iterations):
        node_labels = wl_step(G, node_labels)
        c = Counter()
        # count node labels
        for i,d in enumerate(node_labels):
            h = blake2b(digest_size=digest_size)
            h.update(d.encode('ascii'))
            hexed = h.hexdigest()
            c.update([hexed])
        # sort the counter, extend total counts
        items.extend(sorted(c.items(), key=lambda x: x[0]))
    # hash the final counter
    h = blake2b(digest_size=digest_size)
    h.update((str(tuple(items))+str(graph_property)).encode('ascii'))
    h = h.hexdigest()
    return int(h,16)

def test_me():
    G1 = Graph()
    v1 = list(G1.add_vertex(4))
    G1.add_edge(v1[0],v1[1])
    G1.add_edge(v1[1],v1[3])
    G1.add_edge(v1[2],v1[3])
    G1.add_edge(v1[0],v1[3])
    owner1 = G1.new_vertex_property("short")
    G1.vp.o = owner1
    owner1.a = np.array([0,0,0,0])
    hashval = G1.new_vertex_property("long")
    G1.vp.h = hashval
    G1.gp["h"] = G1.new_graph_property("long")
    G1.gp["b"] = G1.new_graph_property("bool")


    G2 = Graph()
    v2 = list(G2.add_vertex(4))
    G2.add_edge(v2[0],v2[1])
    G2.add_edge(v2[1],v2[2])
    G2.add_edge(v2[2],v2[0])
    G2.add_edge(v2[2],v2[3])
    owner2 = G2.new_vertex_property("short")
    G2.vp.o = owner2
    owner2.a = np.array([0,0,0,0])
    hashval = G2.new_vertex_property("long")
    G2.vp.h = hashval
    G2.gp["h"] = G2.new_graph_property("long")
    G2.gp["b"] = G2.new_graph_property("bool")

    s = perf_counter()
    wl_hash(G1,owner1)
    wl_hash(G2,owner2)
    print(f"time per hash: {(perf_counter()-s)/2}")

    print(G1.vp.h.get_array())
    print(G2.vp.h.get_array())
    print(G1.gp["h"],G2.gp["h"])
    graph_draw(G1, vertex_text=G1.vertex_index, output="G1.pdf")
    graph_draw(G2, vertex_text=G2.vertex_index, output="G2.pdf")

if __name__ == "__main__":
    test_me()
