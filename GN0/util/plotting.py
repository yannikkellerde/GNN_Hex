"""File to create the results plots from the thesis.

Key functions include:
    elo_plot_curriculum: Create Figure 5.6 a
    plot_transfer_elos: Create Figure 5.3
    plot_first_move_plot: Create Figure 5.2
"""

from GN0.RainbowDQN.evaluate_elo import Elo_handler,random_player
from GN0.util.convert_graph import convert_node_switching_game
import torch
import numpy as np
from graph_game.graph_tools_games import Hex_game
from GN0.RainbowDQN.Rainbow.common.utils import get_highest_model_path
from GN0.models import get_pre_defined
import matplotlib.pyplot as plt
import math
import json
import os

basepath = os.path.abspath(os.path.dirname(__file__))
device = "cpu"

def elo_plot_curriculum():
    # cuts = [1706,4500,8120,12500,17500]
    # cuts = np.array([3222680,8490240,15288960,23495040,32868480])/1000000
    cuts = []
    folder_path = os.path.join(basepath,"elo_tables")
    total_min = 0
    total_max = 0
    scat_points = []
    for i in range(5,12):
        out = []
        file_path = os.path.join(folder_path,f"{i}.json")
        with open(file_path,"r") as f:
            data = json.load(f)
        if i == 10:
            with open(os.path.join(folder_path,f"{i}_fillin.json"),"r") as f:
                data["data"].extend(json.load(f)["data"])
        for el in data["data"]:
            if el[0] not in ("random","old_model"):
                out.append([int(el[0].split("_")[0])/1000000,int(el[1])])
        data = np.array(list(sorted(out,key=lambda x:x[0])))
        if i!=5:
            cuts.append(np.min(data[:,0]))
        total_min = min(np.min(data[:,1]),total_min)
        total_max = max(np.max(data[:,1]),total_max)
        scat_points.append((data[0,0],data[0,1]))
        plt.plot(data[:,0],data[:,1],color='#1f77b4')
    scat_points = np.array(scat_points)
    plt.scatter(scat_points[:,0],scat_points[:,1],color="red",marker="x")
    plt.xlabel("game frame (millions)")
    plt.ylabel("elo")
    plt.vlines(x=cuts, ymin=total_min-5, ymax=total_max+5, color='black',linestyles="dashed")
    plt.hlines(y=0, xmin=-1, xmax=np.max(data[:,0]), color='black',linestyles="dashed")
    plt.gcf().tight_layout()
    plt.margins(0)
    plt.show()


def plot_transfer_elos(size=7):
    e = Elo_handler(size,k=2,device=device)
    if size==7:
        e.load_a_model_player(get_highest_model_path("gnn_7x7/7"),"modern_two_headed","gnn")
        e.load_a_model_player(get_highest_model_path("cnn_7x7_fully_conv/7"),"fully_conv","cnn",cnn_mode=True,cnn_hex_size=None,cnn_zero_fill=True)
        start = 5
        end = 14
    elif size==11:
        e.load_a_model_player(get_highest_model_path("rainbow_gnn_11x11/11"),"modern_two_headed","gnn")
        # e.load_a_model_player(get_highest_model_path("rainbow_cnn_11x11/11"),"unet","cnn",cnn_mode=True,cnn_hex_size=None)
        e.load_a_model_player(get_highest_model_path("astral-haze-209/11"),"gao","cnn",cnn_mode=True,cnn_hex_size=None,gao_style=True,border_fill=True)
        # e.load_a_model_player(get_highest_model_path("absurd-paper-213/11"),"gao","cnn",cnn_mode=True,cnn_hex_size=None,gao_style=True, border_fill=False)
        start = 8
        end = 16
    else:
        raise ValueError("size unknown")
    e.add_player(name="random",model=random_player,set_rating=0,uses_empty_model=False,simple=True,rating_fixed=True)

    gnn_win_percents=[]

    for hex_size in range(start,end):
        e.size = hex_size
        e.players["gnn"]["rating"]=0
        e.players["cnn"]["rating"]=0
        # e.roundrobin(3,None,must_include_players=["random","gnn","cnn"],score_as_n_games=5000)
        result1 = e.play_some_games("cnn","gnn",None,0,progress=True)
        result2 = e.play_some_games("gnn","cnn",None,0,progress=True)
        cnn_wins = result1["cnn"]+result2["cnn"]
        gnn_wins = result1["gnn"]+result2["gnn"]
        gnn_win_percents.append(gnn_wins/(cnn_wins+gnn_wins))

    gnn_win_percents = np.array(gnn_win_percents)*100
    plt.yticks((0,50,100))
    plt.xticks(np.arange(start,end))
    plt.ylabel("Win percent")
    plt.xlabel("Hex size")

    plt.bar(np.arange(start,end),gnn_win_percents,width=0.8,color="darkorange",label="GNN")
    plt.bar(np.arange(start,end),100-gnn_win_percents,width=0.8,bottom=gnn_win_percents, color="b",label="CNN")
    plt.legend()
    plt.show()

def plot_first_move_plot(model_identifier,model_path,cnn_mode,hex_size):
    new_game = Hex_game(hex_size)
    stuff = torch.load(model_path,map_location="cpu")
    model = get_pre_defined(model_identifier,args=stuff["args"])
    model.load_state_dict(stuff["state_dict"])
    if cnn_mode:
        to_pred_maker = new_game.board.to_input_planes().unsqueeze(0).to(device)
        new_game.view.gp["m"] = not new_game.view.gp["m"]
        to_pred_breaker = new_game.board.to_input_planes().unsqueeze(0).to(device)
        pred_maker = model(to_pred_maker).squeeze()
        pred_breaker = model(to_pred_breaker).squeeze()
        maker_vinds = {new_game.board.board_index_to_vertex_index[int(i)]:value for i,value in enumerate(pred_maker)}
        breaker_vinds = {new_game.board.board_index_to_vertex_index[int(i)]:value for i,value in enumerate(pred_breaker)}
    else:
        to_pred_maker = convert_node_switching_game(new_game.view,global_input_properties=[1],need_backmap=True,old_style=True).to(device)
        to_pred_breaker = convert_node_switching_game(new_game.view,global_input_properties=[0],need_backmap=True,old_style=True).to(device)
        pred_maker = model(to_pred_maker.x,to_pred_maker.edge_index).squeeze()
        pred_breaker = model(to_pred_breaker.x,to_pred_breaker.edge_index).squeeze()
        maker_vinds = {to_pred_maker.backmap[int(i)]:value for i,value in enumerate(pred_maker) if int(i)>1}
        breaker_vinds = {to_pred_breaker.backmap[int(i)]:value for i,value in enumerate(pred_breaker) if int(i)>1}
    maker_vprop = new_game.view.new_vertex_property("float")
    breaker_vprop = new_game.view.new_vertex_property("float")
    for key,value in maker_vinds.items():
        maker_vprop[new_game.view.vertex(key)] = value

    for key,value in breaker_vinds.items():
        breaker_vprop[new_game.view.vertex(key)] = value
    plt.cla()
    fig_maker = new_game.board.matplotlib_me(vprop=maker_vprop,color_based_on_vprop=True,fig=plt.gcf())
    plt.show()
    plt.cla()
    fig_breaker = new_game.board.matplotlib_me(vprop=breaker_vprop,color_based_on_vprop=True,fig=plt.gcf())
    plt.show()




if __name__ == "__main__":
    plot_transfer_elos(11)
    # plot_transfer_elos(7)
    # plot_first_move_plot("modern_two_headed","../RainbowDQN/Rainbow/checkpoints/gnn_7x7/7/checkpoint_14395392.pt",False,7)
    # plot_first_move_plot("fully_conv","../RainbowDQN/Rainbow/checkpoints/cnn_7x7_fully_conv/7/checkpoint_28790784.pt",True,7)
    # elo_plot_curriculum()
