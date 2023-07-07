"""Use the interactive_hex_window to play against a neural network model"""

from GN0.models import get_pre_defined
from GN0.util.convert_graph import convert_node_switching_game
from graph_game.graph_tools_games import Hex_game
import os
import torch
from graph_game.hex_gui import playerify_advantage_model,interactive_hex_window, playerify_maker_breaker, maker_breaker_evaluater,advantage_model_to_evaluater, make_board_chooser, make_responding_evaluater
from argparse import Namespace
from GN0.RainbowDQN.Rainbow.common.utils import get_highest_model_path
from GN0.alpha_zero.NN_interface import NNetWrapper


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
basepath = os.path.abspath(os.path.dirname(__file__))

def play_in_gui():
    # version = 11040000
    version = None
    # path = get_highest_model_path("daily-totem-131")
    # path = get_highest_model_path("azure-snowball-157")
    # path = get_highest_model_path("misty-firebrand-26/5")
    # path = os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/gnn_7x7/7/checkpoint_14395392.pt")
    cnn_mode = True
    gao_mode = True
    # path = os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/cnn_5x5_fully_conv/5/checkpoint_40187136.pt")
    # path = get_highest_model_path("gnn_7x7/7")
    path = get_highest_model_path("astral-haze-209/11")
    # path = get_highest_model_path("beaming-firecracker-2201/11")
    # path = "../alpha_zero/checkpoints/181.pt"
    # path = get_highest_model_path("breezy-morning-37")
    if version is not None:
        path = os.path.join(os.path.dirname(path),f"checkpoint_{version}.pt")
    stuff = torch.load(path,map_location=device)
    args = stuff["args"]
    print(args)
    if args is None:
        args = Namespace(num_layers=8,head_layers=2,hidden_channels=25)
    # model = get_pre_defined("policy_value",args).to(device)
    model = get_pre_defined("gao",args).to(device)
    # model = get_pre_defined("fully_conv",args).to(device)

    model.load_state_dict(stuff["state_dict"])
    if "cache" in stuff and stuff["cache"] is not None:
        model.import_norm_cache(*stuff["cache"])
    model.eval()

    # wrap = NNetWrapper(model,device=device)
    # player = make_board_chooser(wrap.choose_move)
    # evaluater = make_responding_evaluater(wrap.be_evaluater)

    player = playerify_advantage_model(model,cnn_mode=cnn_mode,gao_mode=gao_mode)
    evaluater = advantage_model_to_evaluater(model,cnn_mode=cnn_mode,gao_mode=gao_mode)
    interactive_hex_window(11,model_player=player,model_evaluater=evaluater,device_=device)


def play_vs_old():
    breaker = get_pre_defined("sage+norm").to(device)
    maker = get_pre_defined("sage+norm").to(device)
    stuff_breaker = torch.load("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_breaker_32800000.pt",map_location=device)

    breaker.load_state_dict(stuff_breaker["state_dict"])
    if "cache" in stuff_breaker and stuff_breaker["cache"] is not None:
        breaker.import_norm_cache(*stuff_breaker["cache"])
    breaker.eval()
    stuff_maker = torch.load("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_maker_32800000.pt",map_location=device)
    maker.load_state_dict(stuff_maker["state_dict"])
    if "cache" in stuff_maker and stuff_maker["cache"] is not None:
        maker.import_norm_cache(*stuff_maker["cache"])
    maker.eval()
    player = playerify_maker_breaker(maker,breaker)
    evaluater = maker_breaker_evaluater(maker,breaker)
    interactive_hex_window(11,model_player=player,model_evaluater=evaluater)

if __name__ == "__main__":
    # play_vs_model()
    # interactive_hex_window(11)
    play_in_gui()
    # play_vs_old()
