import torch
import sys,os
from graph_tool.all import Graph, graph_draw
import numpy as np

HEX_SIZE = 5

def get_position_from_board_index(board_index:int):
    scale = 400/HEX_SIZE
    yend = np.sqrt(3/4)*(HEX_SIZE-1)*scale
    i = board_index//HEX_SIZE
    j = board_index%HEX_SIZE

    coords = [(0.5*j+i)*scale,yend-(np.sqrt(3/4)*j)*scale]
    return coords

def load_data(folder):
    files_tensor_list = ("node_features","edge_indices","policy","board_indices")
    files_tensor = ("value","best_q","game_start_ptr","plys")
    out = {}
    for fname in files_tensor:
        out[fname] = next(torch.jit.load(os.path.join(folder,fname+".pt")).parameters())
    for fname in files_tensor_list:
        out[fname] = list(torch.jit.load(os.path.join(fname+".pt")).parameters())
    return out

def visualize_data(data):
    g = Graph()
    color = g.new_vertex_property("vector<float>")
    position = g.new_vertex_property("vector<double>")
    shape = g.new_vertex_property("string")
    text = g.new_vertex_property("string")
    

    for feat,bi,pi in zip(data["node_features"],data["board_indices"],data["policy"]):
        v = g.add_vertex()
        if feat[0] == 1:
            if feat[1] == 1:
                color[v] = (1,1,0,1)
            else:
                color[v] = (1,0,0,1)
        elif feat[1] == 1:
            color[v] = (0,1,0,1)
        else:
            color[v] = (0,0,0,1)
        position[v] = get_position_from_board_index(bi);
        shape[v] = "hexagon"
        text[v] = f"{pi:.3f}"

    for i in range(data["edge_indices"].size(1)):
        g.add_edge(data["edge_indices"][0][i],data["edge_indices"][1][i])

    vprops = {"color":color,"shape":shape}

    graph_draw(g, pos=position, vprops=vprops, vertex_text=text, output="data_graph.pdf")
