from GN0.models import get_pre_defined
from GN0.convert_graph import convert_node_switching_game
from graph_game.graph_tools_games import Hex_game
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def play_vs_model():
    breaker = get_pre_defined("sage+norm").to(device)
    maker = get_pre_defined("sage+norm").to(device)
    stuff_breaker = torch.load("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/ethereal-glitter-22/checkpoint_breaker_4800000.pt")

    breaker.load_state_dict(stuff_breaker["state_dict"])
    if "cache" in stuff_breaker and stuff_breaker["cache"] is not None:
        breaker.import_norm_cache(*stuff_breaker["cache"])
    breaker.eval()
    stuff_maker = torch.load("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/ethereal-glitter-22/checkpoint_maker_4800000.pt")
    maker.load_state_dict(stuff_maker["state_dict"])
    if "cache" in stuff_maker and stuff_maker["cache"] is not None:
        maker.import_norm_cache(*stuff_maker["cache"])
    maker.eval()
    size = 11
    game = Hex_game(size)
    game.board_callback = game.board.graph_callback

    letters = "abcdefghijklmnopqrstuvwxyz"
    print(game.board.draw_me(green=True))
    while 1:

        move_str = input()
        if move_str == "redraw":
            os.system("pkill mupdf")
            continue
        move = letters.index(move_str[0])+(int(move_str[1:])-1)*size
        game.board.make_move(move,remove_dead_and_captured=True)
        winner = game.who_won()
        print(game.board.draw_me(green=True))
        if winner is not None:
            game = Hex_game(size)
            game.board_callback = game.board.graph_callback
            print(game.board.draw_me(green=True))

        data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(device)
        if game.onturn == "b":
            res = breaker(data.x,data.edge_index).squeeze()
        elif game.onturn == "m":
            res = maker(data.x,data.edge_index).squeeze()
        something = {letters[game.board.vertex_to_board_index[game.view.vertex(data.backmap[i].item())]%size]+str(game.board.vertex_to_board_index[game.view.vertex(data.backmap[i].item())]//size+1):value.item() for i,value in enumerate(res) if i>1}
        for key,value in something.items():
            print(f"{key}:{value:3f}")

        raw_move = torch.argmax(res[2:]).item()+2

        move = data.backmap[raw_move].item()
        game.make_move(move,remove_dead_and_captured=True)
        game.board.onturn = "r" if game.board.onturn=="b" else "b"
        print(game.board.draw_me(green=True))
        winner = game.who_won()
        if winner is not None:
            game = Hex_game(size)
            game.board_callback = game.board.graph_callback
            print(game.board.draw_me(green=True))

if __name__ == "__main__":
    play_vs_model()
