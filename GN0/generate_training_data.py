from graph_game.graph_tools_games import Qango6x6
from graph_game.graph_tools_game import Graph_Store, Graph_game
from GN0.convert_graph import convert_graph
import random
import time

def generate_graphs(number_to_generate):
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
    for i in range(number_to_generate):
        win = False
        while not win:
            actions = game.get_actions(filter_superseeded=False,none_for_win=False)
            win = game.make_move(random.choice(actions))
            game.hashme()
            if win:
                reload(game,start_storage)
                continue
            if game.hash in known_hashes:
                continue
            else:
                known_hashes.add(game.hash)
            moves = game.get_actions()
            evals = game.board.check_move_val(moves,priorize_sets=False)
            for move,ev in zip(moves,evals):
                if (ev in [-3,-4] and game.onturn=="w") or (ev in [3,4] and game.onturn=="b"):
                    game.graph.vp.w[game.graph.vertex[move]] = [True,False]
                elif (ev in [-3,-4] and game.onturn=="b") or (ev in [3,4] and game.onturn=="w"):
                    game.graph.vp.w[game.graph.vertex[move]] = [False,True]
                else:
                    game.graph.vp.w[game.graph.vertex[move]] = [False,False]
            