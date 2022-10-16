from graph_tool.all import Graph,Vertex,graph_draw,radial_tree_layout
from GN0.alpha_zero.MCTS import MCTS as MCTS_old,Node,Leafnode,upper_confidence_bound
from GN0.alpha_zero.MCTS_new import MCTS
from GN0.alpha_zero.NN_interface import NNetWrapper
from graph_game.shannon_node_switching_game import Node_switching_game
from graph_game.graph_tools_games import get_graph_only_hex_game,Hex_game
import numpy as np
from typing import Union
import os
import time
from GN0.models import get_pre_defined
from argparse import Namespace
from graph_game.graph_tools_hashing import get_unique_hash
import torch

def dummy_nn(game:Node_switching_game):
    moves = game.get_actions()
    prob = np.array(list(range(len(moves))),dtype=float)+1
    prob/=np.sum(prob)
    value = 0.7 if len(moves)%2==0 else 0.3
    return moves,torch.from_numpy(prob),torch.tensor([value])

def graph_state(game,prob=None):
    if prob is None:
        prior_map = None
    else:
        prior_map = game.view.new_vertex_property("double")
        for i,vertex in enumerate(game.get_actions()):
            prior_map[vertex] = prob[i]
    os.system("pkill -f 'mupdf node_state.pdf'")
    game.draw_me(fname="node_state.pdf",vprop1=prior_map,decimal_places=3)
    os.system("nohup mupdf node_state.pdf > /dev/null 2>&1 &")


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

def graph_from_hash_mcts(mcts:MCTS,root:Node_switching_game,to_show="num_visits",last_hash=int):
    def recursive_add(state:Graph,cur_vertex:Vertex):
        game.set_to_graph(Graph(state))
        s = get_unique_hash(game.view)
        if s not in mcts.Ps:
            return
        actions = game.get_actions()
        if s in mcts.Qsa:
            ucb_many = mcts.Qsa[s] + mcts.cpuct * mcts.Ps[s] * np.sqrt(mcts.Ns[s]) / (1 + mcts.Nsa[s])
        else:
            ucb_many = 0.5 + mcts.cpuct * mcts.Ps[s] * np.sqrt(mcts.Ns[s])
        nsa = mcts.Nsa[s] if s in mcts.Nsa else np.zeros_like(mcts.Ps[s])
        qsa = mcts.Qsa[s] if s in mcts.Qsa else np.ones_like(mcts.Ps[s])*0.5
        for ucb,prior,visits,q,action in zip(ucb_many,mcts.Ps[s],nsa,qsa,game.get_actions()):
            game.set_to_graph(Graph(state))
            game.make_move(action)
            new_state = Graph(game.graph)
            new_s = get_unique_hash(game.view)
            v = g.add_vertex()
            g.add_edge(cur_vertex,v)
            color[v] = "blue" if state.gp["m"] else "red"
            if to_show=="num_visits":
                text[v] = str(visits)
            elif to_show == "q":
                text[v] = f"{q:.2f}"[1:] if q<1 else "1"
            elif to_show == "prior":
                text[v] = f"{prior:.2f}"
            elif to_show == "m":
                text[v] = str(action)
            elif to_show=="numbers":
                text[v] = str(int(v))
            elif to_show=="ucb":
                text[v] = f"{float(ucb):.2f}"

            number_to_state[int(v)] = new_state
            if last_hash==new_s:
                halo_color[v] = [0,1,0,0.5]
                weather_halo[v] = True
            else:
                weather_halo[v] = False
                halo_color[v] = [0,0,0,0]
            if new_s in mcts.Ps:
                shape[v] = "circle"
                recursive_add(new_state,v)
            else:
                shape[v] = "square"

    state = Graph(root.graph)
    game = root

    number_to_state = {}
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

    root_hash = get_unique_hash(game.view)
    number_to_state[0] = Graph(state)

    v = g.add_vertex()
    halo_color[v] = [0,0,0,0]
    if root_hash==last_hash:
        halo_color[v] = [0,1,0,0.5]
        weather_halo[v] = True

    recursive_add(state,v)

    halo_size.a = 1.1
    makerturn = state.gp["m"]
    color[v] = "red" if makerturn else "blue"
    shape[v] = "circle" if root_hash in mcts.Ps else "square"
    text[v] = ""
    size.a = 25
    return g,number_to_state
        


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


def visualize_MCTS(new_version=True):
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

    if new_version:
        mcts = MCTS(game.copy(),dummy_nn,args=Namespace(cpuct=1),remove_dead_and_captured=False)
        last_hash = get_unique_hash(game.view)
    else:
        mcts = MCTS_old(game,nn.predict_for_mcts,remove_dead_captured=False)
        last_node = mcts.root
    mode = "num_visits"
    while 1:
        if new_version:
            g,number_to_state = graph_from_hash_mcts(mcts,game.copy(),to_show=mode,last_hash=last_hash)
        else:
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
                if new_version:
                    moves,probs = mcts.extract_result(Graph(game.graph),temp=0)
                else:
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
                if new_version:
                    moves,probs = mcts.extract_result(Graph(game.graph),temp=0)
                    graph_state(game.copy(),probs)
                else:
                    moves,probs = mcts.extract_result(0)
                    graph_node(mcts.root,probs)
                print(moves,probs)

            elif command=="":
                if new_version:
                    value = mcts.single_iteration(Graph(game.graph))
                    g = Node_switching_game.from_graph(mcts.leaf_graph)
                    s = get_unique_hash(g.view)
                    if s in mcts.Ps:
                        graph_state(g,mcts.Ps[s])
                    else:
                        graph_state(g)
                    last_hash = s
                else:
                    leaf,value = mcts.single_iteration()
                    if isinstance(leaf,Node):
                        graph_node(leaf)
                    if mcts.done:
                        print(mcts.extract_result(1))
                        print("MCTS is done")
                print(f"got value {value}")
                break
            else:
                if new_version:
                    state = number_to_state[int(command)]
                    g = Node_switching_game.from_graph(state)
                    graph_state(g,mcts.Ps[get_unique_hash(g.view)])
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
