from GN0.generate_training_data import generate_graphs
from GN0.model_frontend import evaluate_game_state
from GN0.GCN import GCN
from graph_game.graph_tools_games import Qango6x6
from GN0.convert_graph import convert_graph
import random
import torch

def test_random_game(model:torch.nn.Module):
    game = Qango6x6()
    start_pos = list("ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff")
    game.board.position = start_pos
    game.graph_from_board()
    iswin = game.graph.new_vertex_property("vector<bool>")
    # 1: Is win for the player to move by forced moves
    # 2: Is win for the player not to move by forced moves

    game.graph.vp.w = iswin
    for v in game.graph.vertices():
        game.graph.vp.w[v] = [False] * 2
    graphs = []
    known_hashes = set()
    win = False
    while 1:
        actions = game.get_actions(filter_superseeded=False,none_for_win=False)
        if len(actions) == 0:
            break
        move = random.choice(actions)
        win = game.make_move(move)
        game.hashme()
        if win:
            break
        if game.hash in known_hashes:
            continue
        else:
            known_hashes.add(game.hash)
        moves = game.get_actions(filter_superseeded=False,none_for_win=False)
        if len(moves) == 0:
            break
        for move in moves:
            game.view.vertex(move)
        storage = game.extract_storage()
        evals = game.check_move_val(moves,priorize_sets=False)
        game.load_storage(storage)
        for move,ev in zip(moves,evals):
            if (ev in [-3,-4] and game.onturn=="w") or (ev in [3,4] and game.onturn=="b"):
                game.graph.vp.w[game.view.vertex(move)] = [True,False]
            elif (ev in [-3,-4] and game.onturn=="b") or (ev in [3,4] and game.onturn=="w"):
                game.graph.vp.w[game.view.vertex(move)] = [False,True]
            else:
                game.graph.vp.w[game.view.vertex(move)] = [False,False]
        game.board.position = game.board.pos_from_graph()
        #game.board.draw_me_with_prediction(game.graph.vp.w)
        evaluate_game_state(model,game)
        model_pred = game.board.draw_me_with_prediction(game.view.vp.p)
        ground_truth = game.board.draw_me_with_prediction(game.view.vp.w)
        mplines = model_pred.split("\n")
        gtlines = ground_truth.split("\n")
        out_str = "\n".join(["\t".join(x) for x in zip(mplines,gtlines)])
        print(out_str)
    return graphs

if __name__ == "__main__":
    device = "cpu"
    model = GCN(3,2,conv_layers=8,conv_dim=64).to(device)
    model.load_state_dict(torch.load("../model/GCN_model.pt"))
    graphs = test_random_game(model)