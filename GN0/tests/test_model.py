from graph_game.graph_tools_games import Qango6x6
from graph_game.winpattern_game import Graph_Store, Winpattern_game
from GN0.util.convert_graph import convert_graph
import random
import time
import torch


def test_model(model):
    """ Run a game of Qango and compare the predictions of the model with
    the ground-truth.

    Args:
        model: The model to test.
    """
    def reload(game:Winpattern_game,storage:Graph_Store):
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
    game.board.graph_from_board()
    iswin = game.graph.new_vertex_property("vector<bool>")
    # 1: Is win for the player to move by forced moves
    # 2: Is win for the player not to move by forced moves

    game.graph.vp.w = iswin
    for v in game.graph.vertices():
        game.graph.vp.w[v] = [False] * 2
    start_storage = game.extract_storage()
    graphs = []
    known_hashes = set()
    for _ in trange(games_to_play):
        win = False
        while 1:
            actions = game.get_actions(filter_superseeded=False,none_for_win=False)
            if len(actions) == 0:
                break
            move = random.choice(actions)
            win = game.make_move(move)
            game.board.position = game.board.pos_from_graph()
            game.board.draw_me()
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
            graphs.append(convert_graph(game.view)[0])
        reload(game,start_storage)
