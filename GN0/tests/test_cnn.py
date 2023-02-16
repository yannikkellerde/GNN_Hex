from graph_game.hex_board_game import Hex_board
from graph_game.graph_tools_games import Hex_game
from GN0.models import get_pre_defined
from GN0.util.util import count_model_parameters
from collections import namedtuple
from argparse import Namespace

def env_test(model,hex_size=7):
    g = Hex_game(hex_size)
    g.board_callback = g.board.graph_callback
    print(g.board.to_input_planes())
    print(g.board.to_input_planes(hex_size))
    g.make_move(hex_size)
    print(g.board.to_input_planes())
    print(g.board.to_input_planes(hex_size))
    print(model(g.board.to_input_planes(hex_size).unsqueeze(0)))

def param_counting():
    args = Namespace(**{
        "num_layers":30,
        "cnn_hex_size":8,
        "cnn_head_filters":2,
        "cnn_body_filters":37,
        "num_head_layers":1,
        }
    )
    cnn_model = get_pre_defined("fully_conv",args=args)
    env_test(cnn_model)
    env_test(cnn_model, hex_size=9)
    args = Namespace(**{
        "num_layers":30,
        "hidden_channels":70,
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
    param_counting()
