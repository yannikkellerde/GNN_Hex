from GN0.util.convert_graph import convert_node_switching_game_back
from graph_game.shannon_node_switching_game import Node_switching_game
from graph_tool.all import Graph
import os
import time

def visualize_transitions(transitions):
    for i,(state,action,reward,next_state,done) in enumerate(transitions):
        graph_before:Graph = convert_node_switching_game_back(state)
        graph_after:Graph = convert_node_switching_game_back(next_state)
        game_before = Node_switching_game.from_graph(graph_before)
        game_after = Node_switching_game.from_graph(graph_after)
        print(f"\nTransition {i}")
        print(f"Action {action} was chosen")
        print(f"Onturn changed from {game_before.onturn} to {game_after.onturn}")
        print(f"A total reward of {reward} was gained")
        print(f"Number of nodes changed from {graph_before.num_vertices()} to {graph_after.num_vertices()}")
        if done:
            print("Game is done")
        else:
            print("Game is not done")
        print("")
        game_before.draw_me("game_before.pdf")
        game_after.draw_me("game_after.pdf")
        while 1:
            todo = input()
            if todo == "":
                break
            elif todo == "b":
                os.system("nohup mupdf game_before.pdf > /dev/null 2>&1 &")
                time.sleep(0.1)
                os.system("bspc node -f west")
            elif todo == "a":
                os.system("nohup mupdf game_after.pdf > /dev/null 2>&1 &")
                time.sleep(0.1)
                os.system("bspc node -f west")
            elif todo == "k":
                os.system("pkill mupdf")
                
        os.system("pkill mupdf")


