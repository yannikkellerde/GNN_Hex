""" The idea here is to instead of learning with GN0, we just learn the true
win/loss as a function of the board position."""

from graph_game.graph_tools_games import Qango6x6
from graph_game.graph_tools_game import Graph_Store, Graph_game
from GN0.convert_graph import convert_graph
import random
import time

def generate_graphs(games_to_play):
    """ Generate training graphs for the Graph net to learn from.
    Makes random moves in the game and uses threat search to evaluate
    the all moves in all positions. Then stores the graphs as training
    sets for the Graph net to train on.

    Args:
        games_to_play: Number of games to play.
    """
    def reload(game:Graph_game,storage:Graph_Store):
        game.load_storage(start_storage)
        game.graph_from_board()
        iswin = game.graph.new_vertex_property("vector<bool>")
        game.graph.vp.w = iswin
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
    start_storage = game.extract_storage()
    graphs = []
    known_hashes = set()
    for _ in range(games_to_play):
        for v in game.graph.vertices():
            iswin[v] = [False] * 2
        win = False
        while 1:
            actions = game.get_actions(filter_superseeded=False,none_for_win=False)
            win = game.make_move(random.choice(actions))
            game.hashme()
            if win:
                reload(game,start_storage)
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
            graphs.append(convert_graph(game.view))
    return graphs
