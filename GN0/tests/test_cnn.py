from graph_game.hex_board_game import Hex_board
from graph_game.graph_tools_games import Hex_game
from GN0.models import get_pre_defined
from GN0.util.util import count_model_parameters
from collections import namedtuple
from argparse import Namespace

def env_test():
    g = Hex_game(5)
    g.board_callback = g.board.graph_callback
    print(g.board.to_input_planes())
    print(g.board.to_input_planes(6))
    g.make_move(5)
    print(g.board.to_input_planes())
    print(g.board.to_input_planes(6))
    model = get_pre_defined("cnn_two_headed")
    print(model(g.board.to_input_planes(6).unsqueeze(0)))

def param_counting():
    cnn_model = get_pre_defined("cnn_two_headed")
    args = Namespace(**{
        "num_layers":7,
        "hidden_channels":20,
        "num_head_layers":2,
        "noisy_dqn":False,
        "noisy_sigma0":False,
        "norm":False
        }
    )
    gnn_model = get_pre_defined("modern_two_headed",args)
    print("CNN parameters",count_model_parameters(cnn_model))
    print("GNN parameters",count_model_parameters(gnn_model))

    
    

if __name__ == "__main__":
    # env_test()
    param_counting()
