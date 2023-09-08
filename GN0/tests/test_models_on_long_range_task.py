import torch
from graph_game.graph_tools_games import Hex_game
from GN0.models import get_pre_defined
from GN0.util.convert_graph import convert_node_switching_game
from graph_game.hex_gui import advantage_model_to_evaluater
import matplotlib.pyplot as plt
from GN0.util.util import downsample_cnn_outputs, downsample_gao_outputs
import os

basepath = os.path.abspath(os.path.dirname(__file__))

device = "cpu"

def load_a_model(checkpoint,model_identifier):
    stuff = torch.load(checkpoint,map_location=device)
    model = get_pre_defined(model_identifier,stuff["args"]).to(device)
    model.load_state_dict(stuff["state_dict"])
    return model

def test_on_long_range_tasks(model,cnn_mode,min_hex_size=5,max_hex_size=13,flip=False,gao_mode=False):
    fails = []
    def get_board_outputs():
        if cnn_mode:
            if gao_mode:
                model_input = game.board.to_gao_input_planes().unsqueeze(0)
                board_outputs = downsample_gao_outputs(model(model_input).reshape(model_input.shape[0],-1),hex_size)
                mask = downsample_gao_outputs(torch.logical_or(model_input[:,0].reshape(model_input.shape[0],-1).bool(),model_input[:,1].reshape(model_input.shape[0],-1).bool()),hex_size)
            else:
                model_input = game.board.to_input_planes(flip=flip).unsqueeze(0)
                board_outputs = model(model_input)
                mask = torch.logical_or(model_input[:,0].reshape(model_input.shape[0],-1).bool(),model_input[:,1].reshape(model_input.shape[0],-1).bool())
            board_outputs[mask] = -5
            board_outputs = board_outputs.squeeze()
        else:
            board_outputs = torch.ones(hex_size**2)*(-5)
            vprop = evaluater(game)
            for vertex in game.view.vertices():
                if int(vertex)>1:
                    board_outputs[game.board.vertex_index_to_board_index[vertex]] = vprop[vertex]
        return board_outputs

    def fill_game_with_long_range_position(hex_size,defender_color):
        md = defender_color=="m"
        attacker_color = "m" if defender_color == "b" else "b"
        for i in range(hex_size):
            game.make_move(game.board.board_index_to_vertex[i+hex_size if md else hex_size*i+1],force_color=defender_color,remove_dead_and_captured=False)
            if i > 1:
                game.make_move(game.board.board_index_to_vertex[i*hex_size if md else i],force_color=attacker_color,remove_dead_and_captured=False)
                game.make_move(game.board.board_index_to_vertex[i*hex_size+hex_size-1 if md else i+(hex_size-1)*hex_size],force_color=attacker_color,remove_dead_and_captured=False)
            if i!=0 and i!=hex_size-1:
                game.make_move(game.board.board_index_to_vertex[i if md else hex_size*i],force_color=attacker_color,remove_dead_and_captured=False)
        game.view.gp["m"] = defender_color=="m"

    if not cnn_mode:
        evaluater = advantage_model_to_evaluater(model)
    for hex_size in range(min_hex_size,max_hex_size+1):
        game = Hex_game(hex_size)
        game.board_callback = game.board.graph_callback
        fill_game_with_long_range_position(hex_size,"m")
        # plt.cla()
        # game.board.matplotlib_me()
        # plt.savefig(f"../../images/long_range_patterns/red_negative_{hex_size}.svg")
        board_outputs = get_board_outputs()
        if torch.argmax(board_outputs).item() == 0:
            fails.append(f"{hex_size}_m_false_positive")
        game.make_move(game.board.board_index_to_vertex[hex_size-1],force_color="b",remove_dead_and_captured=False)
        game.make_move(game.board.board_index_to_vertex[hex_size*2+hex_size//2+((hex_size-2)//2)*hex_size],force_color="m",remove_dead_and_captured=False)
        # plt.cla()
        # game.board.matplotlib_me()
        # plt.savefig(f"../../images/long_range_patterns/red_positive_{hex_size}.svg")
        board_outputs = get_board_outputs()
        if torch.argmax(board_outputs).item() != 0:
            fails.append(f"{hex_size}_m_false_negative")

        game = Hex_game(hex_size)
        game.board_callback = game.board.graph_callback
        fill_game_with_long_range_position(hex_size,"b")
        # plt.cla()
        # game.board.matplotlib_me()
        # plt.savefig(f"../../images/long_range_patterns/blue_negative_{hex_size}.svg")
        board_outputs = get_board_outputs()
        if torch.argmax(board_outputs).item() == 0:
            fails.append(f"{hex_size}_b_false_positive")
            # print(board_outputs)
            # print(game.board.draw_me())
            # exit()
        game.make_move(game.board.board_index_to_vertex[(hex_size-1)*hex_size],force_color="m",remove_dead_and_captured=False)
        game.make_move(game.board.board_index_to_vertex[2+hex_size*(hex_size//2)+((hex_size-2)//2)],force_color="b",remove_dead_and_captured=False)
        board_outputs = get_board_outputs()
        if torch.argmax(board_outputs).item() != 0:
            fails.append(f"{hex_size}_b_false_negative")
        # plt.cla()
        # game.board.matplotlib_me()
        # plt.savefig(f"../../images/long_range_patterns/blue_positive_{hex_size}.svg")
    return fails


if __name__ == "__main__":
    # model = load_a_model(os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/cnn20/20/checkpoint_42286464.pt"),"gao")
    # mistakes = test_on_long_range_tasks(model,True,6,25,False,gao_mode=True)
    # print("Mistakes by gao20",mistakes,len(mistakes))
    # model = load_a_model(os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/gnn20/20/checkpoint_25191936.pt"),"modern_two_headed")
    # mistakes = test_on_long_range_tasks(model,False,6,25,False)
    # print("Mistakes by GNN20",mistakes,len(mistakes))
    # exit()
    model = load_a_model(os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/astral-haze-209/11/checkpoint_18294144.pt"),"gao")
    print("Mistakes by gao_baseline",test_on_long_range_tasks(model,True,8,25,False,gao_mode=True))
    # model = load_a_model(os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/gnn_7x7/7/checkpoint_14395392.pt"),"modern_two_headed")
    # print("Mistakes by GNN-S",test_on_long_range_tasks(model,False,8,25,False))
    # model = load_a_model(os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/cnn_7x7_fully_conv/7/checkpoint_37488000.pt"),"fully_conv")
    # print("\nMistakes by FCN-S",test_on_long_range_tasks(model,True,8,25,True))
    model = load_a_model(os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/rainbow_cnn_11x11/11/checkpoint_65978880.pt"),"unet")
    print("\nMistakes by FCN-L",test_on_long_range_tasks(model,True,8,25,False))
    model = load_a_model(os.path.join(basepath,"../RainbowDQN/Rainbow/checkpoints/rainbow_gnn_11x11/11/checkpoint_44085888.pt"),"modern_two_headed")
    print("\nMistakes by GNN-L",test_on_long_range_tasks(model,False,8,25,False))
