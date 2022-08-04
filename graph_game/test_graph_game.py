from graph_game.graph_tools_games import Tic_tac_toe,Qango6x6,Qango7x7,Qango7x7_plus,Json_game,Hex_game
from graph_game.hex_board_game import Hex_board
from graph_game.shannon_node_switching_game import Node_switching_game
import time
from functools import reduce
from GN0.convert_graph import convert_graph
import pickle
import numpy as np
import os

def test_voltages():
    size = 6
    g = Hex_game(size)
    vprop = g.compute_node_voltages_exact()
    dprop = g.compute_voltage_drops(vprop)
    intprop = g.view.new_vertex_property("int")
    intprop.get_array()[:] = np.around(dprop.get_array()[:]).astype(int)
    g.draw_me(fname="voltages.pdf",vprop=intprop)
    os.system("nohup mupdf voltages.pdf > /dev/null 2>&1 &")


def play_hex():
    size = 6
    g = Hex_game(size)
    letters = "abcdefghijklmnopqrstuvwxyz"
    while 1:
        winner = g.who_won()
        if winner=="m":
            print("Maker(red) has won the game")
        elif winner=="b":
            print("Breaker(blue) has won the game")
        print(g.board.draw_me())
        vprop = g.compute_node_voltages_exact()
        # intprop2 = g.view.new_vertex_property("int")
        # intprop2.get_array()[:] = np.around(vprop.get_array()[:]).astype(int)
        # g.draw_me("cur_game_voltages.pdf",vprop=intprop2)
        # os.system("nohup mupdf cur_game_voltages.pdf > /dev/null 2>&1 &")
        dprop = g.compute_voltage_drops(vprop)
        
        intprop = g.view.new_vertex_property("int")
        intprop.get_array()[:] = np.around(dprop.get_array()[:]).astype(int)
        g.draw_me("cur_game.pdf",vprop=intprop)
        os.system("nohup mupdf cur_game.pdf > /dev/null 2>&1 &")
        time.sleep(0.1)
        os.system("bspc node -f west")
        move_str = input()
        move = letters.index(move_str[0])+(int(move_str[1:])-1)*size
        if g.move_wins(g.board.board_index_to_vertex[move]):
            print("Move wins")
        g.board.make_move(move)
        os.system("pkill mupdf")

     

def test_graph_similarity():
    b1 = Hex_board()
    b1.game = Node_switching_game()
    b1.squares = 5*5
    b1.position = list(
        "ffrff"
        "rrrrr"
        "bbbbb"
        "rrrrr"
        "ffrff"
    )
    b2 = Hex_board()
    b2.game = Node_switching_game()
    b2.squares = 5*5
    b2.position = list(
        "ffrff"
        "rrrrr"
        "bbbbb"
        "bbrrr"
        "ffrff"
    )
    b1.graph_from_board(True)
    b2.graph_from_board(True)
    print(b1.draw_me())
    b1.game.draw_me("b1.pdf")
    b2.game.draw_me("b2.pdf")
    print(b2.draw_me())
    print(b1.game.view.num_edges(),b2.game.view.num_edges())
    print(Hex_board.evaluate_graph_similarity(b1.game.view,b2.game.view,b1.vertex_to_board_index,b2.vertex_to_board_index))
    

def test_hex():
    board = Hex_board()
    board.squares = 11*11 
    #board.position = list("ffrff"
    #                      "rrrrr"
    #                      "bbbbb"
    #                      "rrrrr"
    #                      "ffrff")
    #board.position = ["f"]*board.squares
    board.position = list("fffffrfffff"
                          "fffffrfffff"
                          "rrrrrrrrrrr"
                          "bbbbbbbbbbb"
                          "rrrrrrrrrrr"
                          "fffffrfffff"
                          "fffffrfffff"
                          "fffffrfffff"
                          "fffffrfffff"
                          "fffffrfffff"
                          "fffffrfffff"
                          )
                          
    #board.position = list("ffr"
    #                      "fbf"
    #                      "ffr")
    redgraph = True
    print(board.draw_me())
    board.game = Node_switching_game()
    board.graph_from_board(redgraph)
    board.game.draw_me()
    print(board.game.view.vertex_index.copy().fa)
    exit()
    board.position = ["f"]*board.squares
    board.pos_from_graph(redgraph)
    print(board.draw_me())



def test_graph_nets():
    game = Qango6x6()
    game.board.position = list( "ffffwf"
                                "wbwfbf"
                                "ffffff"
                                "ffffwf"
                                "wbwfbf"
                                "ffffwf")
    game.board.onturn = "b"
    game.board.graph_from_board()
    game.hashme()
    game.draw_me(-1)
    gn_graph = convert_graph([game.view,game.view])
    print(gn_graph)

def test_moving():
    game = Tic_tac_toe()
    ind = 0
    while game.graph.num_vertices()>0:
        moves = game.get_actions()
        print(moves)
        if moves==True:
            print("win")
            break
        game.draw_me(ind)
        ind+=1
        game.make_move(moves[0])

def test_board_representation():
    game = Tic_tac_toe()
    game.board.position = list("fbf"
                               "fff"
                               "fff")
    game.board.onturn = "w"
    game.board.graph_from_board()
    game.hashme()
    moves = game.get_actions()
    game.make_move(moves[1])
    game.draw_me(-1)

def test_forced_move_search():
    game = Qango6x6()
    game.board.position = list( "ffffwf"
                                "wbwfbf"
                                "ffffff"
                                "ffffwf"
                                "wbwfbf"
                                "ffffwf")
    game.board.onturn = "b"
    game.board.graph_from_board()
    game.hashme()
    game.draw_me(-1)
    s = time.perf_counter()
    print(game.forced_move_search())
    print(time.perf_counter()-s)

def test_threat_search():
    game = Qango6x6()
    game.board.position = list( "ffffwf"
                                "wbwfbf"
                                "ffffff"
                                "ffffwf"
                                "wbwfbf"
                                "ffffwf")


    game.board.position = list( "ffffff"
                                "ffffff"
                                "fwffff"
                                "ffbwff"
                                "ffwbbf"
                                "ffffbf")
    game.board.onturn = "b"
    game.board.graph_from_board()
    game.draw_me(-1)
    s = time.perf_counter()
    defenses,win,movelines = game.threat_search()
    print(time.perf_counter()-s)
    print(win,len(defenses),defenses)
    board_view = []
    for d in defenses:
        val = game.board.node_map[d]
        board_view.append((val%6,val//6))
    print(board_view)
    not_defs = set(game.board.node_map.keys())-defenses
    board_view = []
    for d in not_defs:
        val = game.board.node_map[d]
        board_view.append((val%6,val//6))
    print(board_view)
    for line in movelines:
        print([d if type(d)==str else (game.board.node_map[d]%6,game.board.node_map[d]//6) for d in line[1:]])

def test_win_thread_search():
    game = Qango6x6()
    game.board.position = list( "ffffwf"
                                "wbwfbf"
                                "ffffff"
                                "ffffwf"
                                "wbwfbf"
                                "ffffwf")

    """game.board.position = list( "ffffff"
                                "ffffff"
                                "fwffff"
                                "ffbwff"
                                "ffwbff"
                                "ffffbf")"""
    game.board.onturn = "b"
    game.board.graph_from_board()
    s = time.perf_counter()
    winmoves = game.win_threat_search(one_is_enough=False)
    print(time.perf_counter()-s)
    print([(game.board.node_map[x]%6,game.board.node_map[x]//6) for x in winmoves])

def test_pos_from_graph():
    game = Qango6x6()
    game.board.position = list( "ffffff"
                                "ffffff"
                                "ffffff"
                                "ffffff"
                                "fffwff"
                                "fffbff")
    game.board.onturn = "b"
    game.board.graph_from_board()
    game.hashme()
    print(game.hash)
    game.board.inv_maps()
    reco_pos = game.board.pos_from_graph()
    game.board.draw_me(reco_pos)
    game.make_move(game.board.node_map_rev[3*6+3])
    reco_pos = game.board.pos_from_graph()
    game.board.draw_me(reco_pos)
    #game.make_move(list(game.board.node_map.keys())[9])
    #reco_pos = game.board.pos_from_graph()
    #game.board.draw_me(reco_pos)

def display_wsn():
    game = Json_game("json_games/tic_tac_toe.json")
    count = 0
    for wsn in game.board.winsquarenums:
        if len(wsn)!=3:
            continue
        count+=1
        game.board.position = ["f"]*game.board.squares
        for ws in wsn:
            game.board.position[ws] = "W"
        game.board.draw_me()
    print(count)

def test_ai_api():
    ai = Ai_api({"qango6x6":[0,1,2,3],"qango7x7_plus":[2]})
    pos = list( "ffffff"
                "fffbff"
                "ffffwb"
                "ffffff"
                "ffffff"
                "ffffff")

    onturn = "w"
    print(ai.get_move("qango6x6",3,onturn,pos))

def test_json_game():
    game = Json_game("json_games/tic_tac_toe.json")
    game.board.draw_me()

if __name__ == "__main__":
    #test_moving()
    #test_board_representation()
    #test_forced_move_search()
    #test_threat_search()
    #test_pos_from_graph()
    #test_win_thread_search()
    #display_wsn()
    #test_ai_api()
    #test_json_game()
    #test_graph_nets()
    #test_hex()
    play_hex()
    # test_voltages()
    #test_graph_similarity()
