from graph_game.hex_board_game import Hex_board
import torch
from graph_game.graph_tools_games import Hex_game
from GN0.models import get_pre_defined
from GN0.util.util import count_model_parameters
from GN0.torch_script_models import get_current_model, Unet
from collections import namedtuple
from argparse import Namespace
import os

basepath = os.path.abspath(os.path.dirname(__file__))

def env_test(model,hex_size=11):
    g = Hex_game(hex_size)
    g.board_callback = g.board.graph_callback
    print(g.board.to_input_planes())
    print(g.board.to_input_planes(hex_size))
    g.make_move(hex_size)
    print(g.board.to_input_planes())
    print(g.board.to_input_planes(hex_size))
    print(model(g.board.to_input_planes(hex_size).unsqueeze(0)))

def param_counting():
    # model_path = os.path.join(basepath,"..","RainbowDQN/Rainbow/checkpoints/beaming-firecracker-2201/11/checkpoint_91770624.pt")
    model_path = os.path.join(basepath,"..","RainbowDQN/Rainbow/checkpoints/rainbow_gnn_11x11/11/checkpoint_44085888.pt")
    stuff = torch.load(model_path,map_location="cpu")
    print(stuff["args"])
    model = get_pre_defined("modern_two_headed",stuff["args"])
    print("GNN parameters",count_model_parameters(model))
    unet = get_pre_defined("unet")
    print("Unet parameters",count_model_parameters(unet))
    exit()
    # jit_model = torch.jit.load("model_save/mohex_reproduce_large/torch_script_model.pt")
    # args = Namespace(**{
    #     "num_layers":10,
    #     "cnn_hex_size":7,
    #     "cnn_head_filters":1,
    #     "cnn_body_filters":22,
    #     "num_head_layers":1,
    #     }
    # )
    # cnn_model = get_pre_defined("fully_conv",args=args)
    PV_CNN = get_current_model(net_type="PV_CNN")
    print("PV_CNN parameters",count_model_parameters(PV_CNN))
    PV_GNN = get_current_model(net_type="SAGE",hidden_channels=110,hidden_layers=15)
    print("PV_GNN parameters",count_model_parameters(PV_GNN))
    args = Namespace(**{
        "num_layers":15,
        "hidden_channels":110,
        "num_head_layers":2,
        "noisy_dqn":False,
        "noisy_sigma0":False,
        "norm":False
        }
    )
    gnn_model = get_pre_defined("modern_two_headed",args)
    print("Q GNN parameters",count_model_parameters(gnn_model))
    # policy_value_model = get_current_model()
    # unet_model = Unet(3)
    # print("CNN parameters",count_model_parameters(cnn_model))
    # print("GNN parameters",count_model_parameters(gnn_model))
    # print("Unet parameters",count_model_parameters(unet_model))
    # # print("Jit model parameters",count_model_parameters(jit_model))
    # print("Policy value model params",count_model_parameters(policy_value_model))

    
    

if __name__ == "__main__":
    param_counting()
