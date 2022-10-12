from GN0.alpha_zero.replay_buffer import ReplayBuffer
from GN0.util.convert_graph import convert_node_switching_game_back
from graph_tool.all import Graph
from graph_game.shannon_node_switching_game import Node_switching_game
import os
import time

def visualize_data(buffer:ReplayBuffer):
    loader = buffer.get_data_loader(batch_size=1,shuffle=False)
    for data in loader:
        os.system("pkill mupdf")
        graph:Graph = convert_node_switching_game_back(data)
        game = Node_switching_game.from_graph(graph)
        policy_vp = game.view.new_vertex_property("double")
        policy_vp.fa = data.pi.cpu().numpy()
        game.draw_me(fname="show_policy.pdf",vprop1=policy_vp, decimal_places=2, layout="sfdp")
        os.system("nohup mupdf show_policy.pdf > /dev/null 2>&1 &")
        time.sleep(0.1)
        os.system("bspc node -f west")
        print("Value:",data.v.item())
        print("Onturn:",game.onturn)
        input()
        
