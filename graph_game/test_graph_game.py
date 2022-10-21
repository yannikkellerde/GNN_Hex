from graph_tool.all import VertexPropertyMap,GraphView
from graph_game.graph_tools_games import Tic_tac_toe,Qango6x6,Qango7x7,Qango7x7_plus,Json_game,Hex_game,Clique_hex_game
from graph_game.hex_board_game import Hex_board
from graph_game.shannon_node_switching_game import Node_switching_game
import time
from functools import reduce
from GN0.util.convert_graph import convert_node_switching_game,convert_node_switching_game_back
from graph_game.graph_tools_hashing import wl_hash
import pickle
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from alive_progress import alive_bar,alive_it
from time import perf_counter

def test_speed():
    games = [Hex_game(11) for i in range(10)]
    move_time = 0
    start = perf_counter()
    for game in games:
        while game.who_won() is None:
            start_inner = perf_counter()
            game.make_move(random.choice(game.get_actions()),remove_dead_and_captured=False)
            move_time += perf_counter()-start_inner
    print(perf_counter()-start)
    print(move_time)


def test_unique_moves():
    game = Hex_game(11)
    moves = game.board.get_all_unique_starting_moves()
    for move in moves:
        game.board.position[move]="r"
    print(game.board.draw_me())
    

def test_dead_and_captured_consistency():
    size = 11
    for i in alive_it(range(500)):
        simple_game = Hex_game(size)
        fancy_game = Hex_game(size)
        simple_game.board_callback = simple_game.board.graph_callback
        fancy_game.board_callback = fancy_game.board.graph_callback
        while simple_game.who_won() is None:
            move = random.choice(simple_game.get_actions())
            # print("chose",simple_game.board.vertex_index_to_string_move(move))
            simple_game.make_move(move,remove_dead_and_captured=False)
            if move in fancy_game.get_actions():
                if fancy_game.who_won() is None:
                    fancy_game.make_move(move,remove_dead_and_captured=True)
                else:
                    fancy_game.view.gp["m"] = not fancy_game.view.gp["m"]
            else:
                response = fancy_game.get_response(move,for_maker=not fancy_game.view.gp["m"])
                if response is not None:
                    vertex_response = simple_game.board.board_index_to_vertex[response]
                    # print("responded with",simple_game.board.vertex_index_to_string_move(vertex_response))
                    simple_game.make_move(vertex_response,remove_dead_and_captured=False)
                else:
                    # print("No response for",move,"\nResponses:",fancy_game.response_set_breaker if fancy_game.view.gp["m"] else fancy_game.response_set_maker)
                    fancy_game.view.gp["m"] = not fancy_game.view.gp["m"]
            # print("simple\n",simple_game.board.draw_me())
            # print("fancy\n",fancy_game.board.draw_me())
        print(simple_game.who_won(),fancy_game.who_won())
        # print("AAAAAAAAAAAAAAA")
        assert simple_game.who_won()==fancy_game.who_won()

def test_color_consistency():
    size = 11
    g = Hex_game(size)
    g2 = Hex_game(size)
    g.board_callback = g.board.graph_callback
    g2.board_callback = g2.board.graph_callback
    g2.view.gp["m"] = False
    with alive_bar(10000) as bar:
        for i in range(10000):
            actions = g.get_actions()
            actions2 = g2.get_actions()
            # if set(actions)!=set(actions2):
            #     g.board.draw()
            #     g2.board.draw()
            #     return
            a = np.random.choice(actions)
            a2 = g2.board.board_index_to_vertex[g2.board.transpose_move(g.board.vertex_index_to_board_index[a])]
            # if int(a2) not in actions2:
            #     print(g.board.draw_me())
            #     print(g2.board.draw_me())
            #     return
            g.make_move(a,remove_dead_and_captured=True)
            g2.make_move(a2,remove_dead_and_captured=True)
            winner = g.who_won()
            if winner is not None:
                # if g2.who_won() is None or g2.who_won()==winner:
                #     print(g.board.draw_me())
                #     print(g2.board.draw_me())
                #     return
                # print(g.board.draw_me())
                # print(g2.board.draw_me())
                g = Hex_game(size)
                g2 = Hex_game(size)
                g.board_callback = g.board.graph_callback
                g2.board_callback = g2.board.graph_callback
                g2.view.gp["m"] = False
            bar()




def test_iterative_voltages():
    size = 11
    g = Hex_game(size)
    vprop_exact,value_exact = g.compute_node_voltages_exact()
    dprop_exact = g.compute_node_currents(vprop_exact)
    intprop_exact = g.view.new_vertex_property("int")
    intprop_exact.a = np.around(dprop_exact.a).astype(int)
    vprop_approx,value_approx = g.compute_node_voltages_iterate(30)
    dprop_approx = g.compute_node_currents(vprop_approx,check_validity=False)
    intprop_approx = g.view.new_vertex_property("int")
    intprop_approx.a = np.around(dprop_approx.a).astype(int)
    print(f"Exact value {value_exact}. Approximate value {value_approx}")
    g.draw_me(fname="voltages.pdf",vprop1=intprop_exact,vprop2=intprop_approx)
    os.system("nohup mupdf voltages.pdf > /dev/null 2>&1 &")
    assert np.allclose(dprop_approx.a,dprop_exact.a)


def test_voltages():
    size = 11
    g = Hex_game(size)
    vprop,value = g.compute_node_voltages_exact()
    dprop = g.compute_node_currents(vprop)
    intprop = g.view.new_vertex_property("int")
    intprop.a = np.around(dprop.a).astype(int)
    g.board.matplotlib_me(vprop=dprop)
    plt.show()
    # g.draw_me(fname="voltages.pdf",vprop=intprop)
    # os.system("nohup mupdf voltages.pdf > /dev/null 2>&1 &")
    print(value)

def check_conversion_consistency(g:Node_switching_game, vprop:VertexPropertyMap):
    intprop = g.view.new_vertex_property("int")
    intprop.a = np.around(vprop.a).astype(int)
    prev_hash = wl_hash(g.view,intprop,g.view.gp["m"])
    data = convert_node_switching_game(g.view,vprop)
    new_graph,targ_prop_map = convert_node_switching_game_back(data)
    intprop = new_graph.new_vertex_property("int")
    intprop.a = np.around(targ_prop_map.a).astype(int)
    new_hash = wl_hash(new_graph,intprop,new_graph.gp["m"])
    assert prev_hash == new_hash
    return targ_prop_map

def check_hex_pattern(move_list):
    letters = "abcdefghijklmnopqrstuvwxyz"
    size = 6
    g = Hex_game(size)
    g.board_callback = g.board.graph_callback
    for move,color in move_list:
        print(letters[move%size]+str(move//size+1))
        g.board.make_move(move,force_color=color,remove_dead_and_captured=True)
        g.dead_and_captured(iterate=True)
        print(g.board.draw_me())
        g.draw_me("cur_game.pdf")#,vprop=intprop)
        os.system("nohup mupdf cur_game.pdf > /dev/null 2>&1 &")
        time.sleep(0.1)
        os.system("bspc node -f west")
        input()
        os.system("pkill mupdf")

    print(g.board.draw_me())
    g.draw_me("cur_game.pdf")#,vprop=intprop)
    os.system("nohup mupdf cur_game.pdf > /dev/null 2>&1 &")

def check_some_hex_patterns():
    # move_list = [(4*7+1,"r"),(4*7+2,"r"),(3*7+3,"r"),(2*7+4,"r"),(7+3,"b")]
    move_list = [(4*6+4,"r"),(4*6+2,"r"),(4*6,"r"),(6+5,"b"),(2*6+2,"r")]
    # random.shuffle(move_list)
    check_hex_pattern(move_list)


def play_hex():
    size = 9
    g = Hex_game(size)
    g.board_callback = g.board.graph_callback
    letters = "abcdefghijklmnopqrstuvwxyz"
    while 1:
        winner = g.who_won()
        if winner=="m":
            print("Maker(red) has won the game")
        elif winner=="b":
            print("Breaker(blue) has won the game")
        print(g.board.draw_me())
        vprop,value = g.compute_node_voltages_exact()
        dprop = g.compute_node_currents(vprop)
        print(f"board value {value}")
        intprop = g.view.new_vertex_property("int")
        intprop.a = np.around(vprop.a).astype(int)
        g.draw_me("cur_game.pdf",vprop1=intprop)
        os.system("nohup mupdf cur_game.pdf > /dev/null 2>&1 &")
        intprop = check_conversion_consistency(g,intprop)
        time.sleep(0.1)
        os.system("bspc node -f west")
        move_str = input()
        if move_str == "redraw":
            os.system("pkill mupdf")
            continue
        move = letters.index(move_str[0])+(int(move_str[1:])-1)*size
        if g.move_wins(g.board.board_index_to_vertex[move]):
            print("Move wins")
        g.board.make_move(move,remove_dead_and_captured=True)
        g.prune_irrelevant_subgraphs()
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
    # play_hex()
    # check_some_hex_patterns()
    # test_iterative_voltages()
    # test_voltages()
    # test_color_consistency()
    # test_unique_moves()
    test_speed()
    # test_dead_and_captured_consistency()
    #test_graph_similarity()
