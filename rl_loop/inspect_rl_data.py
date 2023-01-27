import torch
import sys,os
from graph_tool.all import Graph, graph_draw
import numpy as np
import time

HEX_SIZE = 7

def get_position_from_board_index(board_index:int):
    scale = 400/HEX_SIZE
    yend = np.sqrt(3/4)*(HEX_SIZE-1)*scale
    i = board_index//HEX_SIZE
    j = board_index%HEX_SIZE

    coords = [(0.5*j+i)*scale,yend-(np.sqrt(3/4)*j)*scale]
    return coords

def load_data(folder):
    files_tensor_list = ("node_features","edge_indices","policy","board_indices")
    files_tensor = ("moves","value","best_q","game_start_ptr","plys")
    out = {}
    for fname in files_tensor:
        out[fname] = next(torch.jit.load(os.path.join(folder,fname+".pt")).parameters())
    for fname in files_tensor_list:
        out[fname] = list(torch.jit.load(os.path.join(folder,fname+".pt")).parameters())
    return out

def visualize_data(data,do_board_index=False):
    g = Graph(directed = False)
    color = g.new_vertex_property("vector<float>")
    position = g.new_vertex_property("vector<double>")
    shape = g.new_vertex_property("string")
    text = g.new_vertex_property("string")
    size = g.new_vertex_property("int")
    

    scale = 400/HEX_SIZE
    yend = np.sqrt(3/4)*(HEX_SIZE-1)*scale
    t1 = g.add_vertex()
    t2 = g.add_vertex()
    shape[t1] = "circle"
    shape[t2] = "circle"
    color[t1] = (0,0,1,1)
    color[t2] = (0,0,1,1)
    size[t1] = 125/HEX_SIZE
    size[t2] = 125/HEX_SIZE
    position[t1] = [0,yend/2]
    position[t2] = [1.5*HEX_SIZE*scale,yend/2]
    for feat,bi,pi in zip(data["node_features"][2:],data["board_indices"][2:],data["policy"]):
        v = g.add_vertex()
        color[v] = (0,0,0,1)
        position[v] = get_position_from_board_index(int(bi));
        shape[v] = "hexagon"
        if do_board_index:
            text[v] = str(int(bi))
        else:
            text[v] = f"{pi:.3f}"
        size[v] = 125/HEX_SIZE

    for i in range(data["edge_indices"].size(1)):
        if data["edge_indices"][0][i]<data["edge_indices"][1][i]:
            g.add_edge(data["edge_indices"][0][i],data["edge_indices"][1][i])

    vprops = {"color":color,"shape":shape,"size":size}

    graph_draw(g, pos=position, vprops=vprops, vertex_text=text, output="data_graph.pdf")

def exploration_loop():
    data = load_data(sys.argv[1])
    idx = 0
    do_board_index = False
    last_idx = 0
    # print(len(data["moves"]), len(data["value"]))
    # assert len(data['moves']) == len(data['value']) == len(data["best_q"])
    while 1:
        os.system("pkill -f 'mupdf data_graph.pdf'")
        visualize_data({key:value[idx] for key,value in data.items() if key!="game_start_ptr"},do_board_index=do_board_index)
        os.system("mupdf data_graph.pdf&")
        time.sleep(0.1)
        os.system("bspc node -f west")
        if len(data['policy'][idx])==len(data["node_features"][idx])-1:
            print(f"Swap prob",data['policy'][idx][-1])
        print(f"Value: {int(data['value'][idx])}")
        print(f"Best Q: {float(data['best_q'][idx])}")
        print(f"Plys to end: {int(data['plys'][idx])}")
        print(f"IDX: {idx}")
        print(f"Next Move: {int(data['moves'][idx])}")
        if last_idx>idx and idx in data['game_start_ptr']-1:
            print("Switched to previous game")
        elif last_idx<idx and idx in data['game_start_ptr']:
            print("Switched to next game")
        todo = input()
        last_idx = idx
        if todo == "b":
            if idx > 0:
                idx-=1
            else:
                print("Reached Rock bottom")
        elif todo == "i":
            do_board_index=not do_board_index
        else:
            if idx < len(data["node_features"]):
                idx+=1
            else:
                print("Reached the top")

if __name__ == "__main__":
    exploration_loop()
