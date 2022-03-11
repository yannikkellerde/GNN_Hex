from model_frontend import evaluate_graph,evaluate_game_state
from GCN import GCN
import torch
from graph_game.graph_tools_game import Graph_game,Graph_Store
from graph_game.graph_tools_games import Qango6x6
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(3,2,conv_layers=8,conv_dim=16,global_dim=16).to(device)

model.load_state_dict(torch.load("model/GCN_model.pt"))
model.eval()

def test_model(games_to_play):
    """ 
    Args:
        games_to_play: Number of games to play.
    """
    def reload(game:Graph_game,storage:Graph_Store):
        game.load_storage(storage)
        iswin = game.graph.new_vertex_property("vector<bool>")
        game.graph.vp.w = iswin
        for v in game.graph.vertices():
            game.graph.vp.w[v] = [False] * 2
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
    start_storage = game.extract_storage()
    graphs = []
    known_hashes = set()
    for _ in range(games_to_play):
        win = False
        while 1:
            actions = game.get_actions(filter_superseeded=False,none_for_win=False)
            if len(actions) == 0:
                break
            move = random.choice(actions)
            win = game.make_move(move)
            game.board.position = game.board.pos_from_graph()
            game.hashme()
            if win:
                break
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
            evaluate_game_state(model,game)
            game.board.draw_me_with_prediction(game.view.vp.p)
            input()
            
        reload(game,start_storage)
    return graphs