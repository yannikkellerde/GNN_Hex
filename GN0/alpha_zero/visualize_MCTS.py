from graph_tool.all import Graph,Vertex,graph_draw,radial_tree_layout
from GN0.alpha_zero.MCTS import MCTS,Node,Leafnode,upper_confidence_bound
from GN0.alpha_zero.NN_interface import NNetWrapper
from graph_game.shannon_node_switching_game import Node_switching_game
from graph_game.graph_tools_games import get_graph_only_hex_game,Hex_game
import numpy as np
from typing import Union
import os
import time
from GN0.models import get_pre_defined
from argparse import Namespace

def dummy_nn(game:Node_switching_game):
    moves = game.get_actions()
    prob = np.array(list(range(len(moves))),dtype=float)+1
    prob/=np.sum(prob)
    value = 0.7 if game.view.gp["m"] else 0.3
    return moves,prob,value

def graph_node(node:Node,prob=None):
    if prob is None:
        prob = node.priors
    game = Node_switching_game.from_graph(node.storage)
    prior_map = game.view.new_vertex_property("double")
    for vertex in game.get_actions():
        prior_map[vertex] = prob[list(node.moves).index(vertex)]

    os.system("pkill -f 'mupdf node_state.pdf'")
    game.draw_me(fname="node_state.pdf",vprop1=prior_map,decimal_places=3)
    os.system("nohup mupdf node_state.pdf > /dev/null 2>&1 &")


def graph_from_root(root:Union[Node,Leafnode],to_show="num_visits",last_node=None):
    def recursive_add(cur_node:Node,cur_vertex:Vertex):
        ucb = upper_confidence_bound(cur_node,1)
        for tv,child,q,visits,one_ucb,prior in zip(cur_node.total_value,cur_node.children,cur_node.Q,cur_node.visits,ucb,cur_node.priors):
            v = g.add_vertex()
            g.add_edge(cur_vertex,v)
            color[v] = "blue" if cur_node.storage.gp["m"] else "red"
            if to_show=="num_visits":
                text[v] = str(visits)
            elif to_show == "q":
                text[v] = f"{q:.2f}"[1:] if q<1 else "1"
            elif to_show == "value":
                if isinstance(child,Node):
                    text[v] = f"{tv:.2f}"
                else:
                    text[v] = child.value
            elif to_show == "prior":
                text[v] = f"{prior:.2f}"
            elif to_show == "m":
                if isinstance(child,Node):
                    text[v] = ""
                else:
                    text[v] = "" if child.move is None else child.move
            elif to_show=="numbers":
                text[v] = str(int(v))
            elif to_show=="ucb":
                text[v] = f"{float(one_ucb):.2f}"

            number_to_node[int(v)] = child
            if last_node is not None and last_node==child:
                halo_color[v] = [0,1,0,0.5]
                weather_halo[v] = True
            else:
                weather_halo[v] = False
                halo_color[v] = [0,0,0,0]
            if isinstance(child,Node):
                shape[v] = "circle"
                recursive_add(child,v)
            else:
                shape[v] = "square"

    number_to_node = {}
    g = Graph(directed=True)
    color = g.new_vertex_property("string")
    shape = g.new_vertex_property("string")
    text = g.new_vertex_property("string")
    size = g.new_vertex_property("int")
    pen_width = g.new_vertex_property("int")
    halo_color = g.new_vertex_property("vector<double>")
    weather_halo = g.new_vertex_property("bool")
    halo_size = g.new_vertex_property("double")


    g.vp.l = halo_size
    g.vp.h = halo_color
    g.vp.g = size
    g.vp.c = color
    g.vp.s = shape
    g.vp.t = text
    g.vp.p = pen_width
    g.vp.b = weather_halo

    v = g.add_vertex()
    halo_color[v] = [0,0,0,0]
    if root==last_node:
        halo_color[v] = [0,1,0,0.5]
        weather_halo[v] = True
    if isinstance(root,Node):
        recursive_add(root,v)

    halo_size.a = 1.1
    makerturn = root.makerturn if isinstance(root,Leafnode) else root.storage.gp["m"]
    color[v] = "red" if makerturn else "blue"
    shape[v] = "square" if isinstance(root,Leafnode) else "circle"
    text[v] = ""
    size.a = 25
    return g,number_to_node


def visualize_MCTS():
    print("""
Instructions:
n: Node number          v: number of visits
q: Q-values             t: Total Values
m: moves                p: Priors
u: ucb                  r: get result
c: select best child for next iteration
[number]: show graph for number.
          """)
    size = 2
    game = get_graph_only_hex_game(size)
    show_game = Hex_game(size)
    # nn = get_pre_defined_mcts_model("misty-firebrand-26/11")
    nnet = get_pre_defined("policy_value",args=Namespace(**{"hidden_channels":25,"num_layers":8,"head_layers":2}))
    nn = NNetWrapper(nnet=nnet)

    mcts = MCTS(game,nn.predict_for_mcts,remove_dead_captured=False)
    mode = "num_visits"
    last_node = mcts.root
    while 1:
        g,number_to_node = graph_from_root(mcts.root,to_show=mode,last_node=last_node)
        graph_draw(g,radial_tree_layout(g,g.vertex(0)),vprops={"halo_size":g.vp.l,"halo":g.vp.b,"halo_color":g.vp.h,"pen_width":g.vp.p,"shape":g.vp.s,"fill_color":g.vp.c,"text":g.vp.t,"size":g.vp.g},bg_color="black",output="mcts.pdf")
        os.system("pkill -f 'mupdf mcts.pdf'")
        os.system("nohup mupdf mcts.pdf > /dev/null 2>&1 &")
        time.sleep(0.1)
        os.system("bspc node -f west")
        while 1:
            command = input()
            if command=="n":
                mode = "numbers"
                break
            elif command=="v":
                mode = "num_visits"
                break
            elif command=="q":
                mode = "q"
                break
            elif command=="t":
                mode = "value"
                break
            elif command=="c":
                moves,probs = mcts.extract_result(0)
                action = moves[np.argmax(probs)]
                child = mcts.root.children[np.argmax(probs)]
                if isinstance(child,Leafnode):
                    print("Failed, child is leafnode")
                else:
                    mcts.next_iter_with_child(action,child.storage)
                    break

            elif command=="m":
                mode = "m"
                break
            elif command=="p":
                mode = "prior"
                break
            elif command=="u":
                mode = "ucb"
                break
            elif command=="r":
                moves,probs = mcts.extract_result(1)
                print(moves,probs)
                graph_node(mcts.root,probs)
            elif command=="":
                leaf,value = mcts.single_iteration()
                last_node=leaf
                if isinstance(leaf,Node):
                    graph_node(leaf)
                print(f"got value {value}")
                if mcts.done:
                    print(mcts.extract_result(1))
                    print("MCTS is done")
                break
            else:
                if int(command)==0:
                    node = mcts.root
                else:
                    node = number_to_node[int(command)]
                if isinstance(node,Node):
                    show_game.set_to_graph(node.storage)
                    show_game.draw_me(fname="uff.pdf")
                    os.system("nohup mupdf uff.pdf > /dev/null 2>&1 &")
                    time.sleep(0.1)
                    os.system("bspc node -f west")
                nd = node.__dict__.copy()
                del nd["parent"]
                del nd["children"]

if __name__=="__main__":
    visualize_MCTS()
