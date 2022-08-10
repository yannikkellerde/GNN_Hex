""" The idea here is to instead of learning with GN0, we just learn the true
win/loss as a function of the board position."""

from graph_game.graph_tools_games import Qango6x6, Hex_game
from graph_game.winpattern_game import Graph_Store, Winpattern_game
from GN0.convert_graph import graph_to_arrays, convert_winpattern_game, convert_node_switching_game
from graph_game.graph_tools_hashing import wl_hash
import random
import time
from tqdm import tqdm,trange
import multiprocessing
import pickle
from typing import Callable

def generate_hex_graphs(games_to_play):
    """ Generate training graphs for the Graph net to learn from.
    Makes random moves in the game and uses voltage drop algorithm
    to evaluate moves. Then stores the graphs as training
    sets for the Graph net to train on.

    Args:
        games_to_play: Number of games to play.
    """
    known_hashes = set()
    graphs = []

    for _ in trange(games_to_play):
        game = Hex_game(15)
        # game.board_callback = game.board.graph_callback
        win = False
        while not win:
            actions = game.get_actions()
            move = random.choice(actions)
            game.make_move(move,remove_dead_and_captured=True)
            hash = wl_hash(game.view,game.view.vp.f)
            # if hash not in known_hashes:
                # known_hashes.add(hash)
                # game.prune_irrelevant_subgraphs()
                # voltprop = game.compute_node_voltages_exact()
                # dropprop = game.compute_voltage_drops(voltprop)
                # data = convert_node_switching_game(game.view,dropprop)
                # graphs.append(data)
            win = game.who_won()
    return graphs

                

def generate_winpattern_game_graphs(games_to_play):
    """ Generate training graphs for the Graph net to learn from.
    Makes random moves in the game and uses threat search to evaluate
    the all moves in all positions. Then stores the graphs as training
    sets for the Graph net to train on.

    Args:
        games_to_play: Number of games to play.
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
            #game.board.draw_me()
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
            graphs.append(convert_winpattern_game(game.view)[0])
        reload(game,start_storage)
    return graphs

def generate_and_store_graphs(generation_func:Callable,games_to_play:int,path:str):
    graphs = generation_func(games_to_play)
    with open(path,"wb") as f:
        pickle.dump(graphs,f)


def generate_graphs_multiprocess(generation_func:Callable,games_to_play:int,paths:str):
    cores = multiprocessing.cpu_count()
    print(len(paths),cores)
    assert len(paths)<=cores
    div,mod = divmod(games_to_play,len(paths))
    params = [div + (1 if x < mod else 0) for x in range(len(paths))]
    partial_generate = lambda *args:generate_and_store_graphs(generation_func,*args)
    with multiprocessing.Pool(len(paths)) as pool:
        pool.starmap(partial_generate,zip(params,paths))

if __name__=="__main__":
    generate_and_store_graphs(generate_hex_graphs,100,"test.pkl")
