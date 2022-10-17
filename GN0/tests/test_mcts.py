from GN0.alpha_zero.MCTS_cached import MCTS as MCTS_cached
from GN0.alpha_zero.MCTS import MCTS as MCTS
from GN0.alpha_zero.MCTS import run_many_mcts
from graph_game.graph_tools_games import Hex_game
from GN0.models import get_pre_defined
from argparse import Namespace
from GN0.alpha_zero.NN_interface import NNetWrapper
from graph_tool.all import Graph
from graph_game.graph_tools_hashing import get_unique_hash
import numpy as np
from graph_game.shannon_node_switching_game import Node_switching_game
import torch
from time import perf_counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device != "cpu"

def dummy_nn(game:Node_switching_game):
    moves = game.get_actions()
    prob = np.array(list(range(len(moves))),dtype=float)+1
    prob/=np.sum(prob)
    value = 0.7 if len(moves)%2==0 else 0.3
    return moves,torch.from_numpy(prob),torch.tensor([value])

def test_batched_speed():
    game = Hex_game(8)
    nnet = get_pre_defined("policy_value",args=Namespace(**{"hidden_channels":50,"num_layers":18,"head_layers":2})).to(device)
    args = Namespace(cpuct=1)
    nn = NNetWrapper(nnet=nnet,device=device)
    mcts = [MCTS(game.copy(),nn=None,args=args,remove_dead_and_captured=True) for _ in range(128)]
    run_many_mcts(mcts,nn=nn.predict_many_for_mcts,num_iterations=200,progress=True)
    for key in mcts[0].timers:
        print(key,sum(sum(m.timers[key]) for m in mcts))

    for key in nn.timers:
        print(key,sum(nn.timers[key]))

def test_batched_correctness():
    game = Hex_game(10)
    nnet = get_pre_defined("policy_value",args=Namespace(**{"hidden_channels":25,"num_layers":8,"head_layers":2}))
    args = Namespace(cpuct=1)
    nn = NNetWrapper(nnet=nnet)
    mcts = MCTS(game.copy(),nn=nn.predict_for_mcts,args=args,remove_dead_and_captured=True)
    mcts.run(graph=Graph(game.graph),num_iterations=200)
    moves1,probs1 = mcts.extract_result(Graph(game.graph),1)

    mcts2 = MCTS(game.copy(),nn=nn.predict_for_mcts,args=args,remove_dead_and_captured=True)
    for i in range(200):
        need_nn = mcts2.find_leaf(set_to_graph=Graph(game.graph))
        if need_nn:
            probs,value = nn.predict_for_mcts(mcts2.game)
            mcts2.process_results(pi=probs,value=value)
        else:
            mcts2.process_results()
    moves2,probs2 = mcts2.extract_result(Graph(game.graph),1)

    mcts3 = MCTS(game.copy(),nn=None,args=args,remove_dead_and_captured=True)
    run_many_mcts([mcts3],nn.predict_many_for_mcts,num_iterations=200)
    moves3,probs3 = mcts3.extract_result(Graph(game.graph),1)

    print(probs1==probs2)
    print(probs2==probs3)

    print(mcts.Nsa[get_unique_hash(game.view)])
    print(mcts2.Nsa[get_unique_hash(game.view)])
    print(mcts3.Nsa[get_unique_hash(game.view)])

def test_mcts():
    game = Hex_game(10)
    nnet = get_pre_defined("policy_value",args=Namespace(**{"hidden_channels":25,"num_layers":8,"head_layers":2}))
    args = Namespace(cpuct=1)
    nn = NNetWrapper(nnet=nnet)
    mcts_cached = MCTS_cached(game.copy(),NN=nn.predict_for_mcts)
    mcts = MCTS(game.copy(),nn=nn.predict_for_mcts,args=args,remove_dead_and_captured=True)

    start = perf_counter()
    mcts_cached.run(iterations=200)
    moves1,probs1 = mcts_cached.extract_result(1)
    standard_time = perf_counter()-start
    start = perf_counter()
    mcts.run(graph=Graph(game.graph),num_iterations=200)
    moves2,probs2 = mcts.extract_result(Graph(game.graph),1)
    new_time = perf_counter()-start

    print("old time",standard_time)
    print("new time",new_time)
    for key,value in mcts.timers.items():
        print(key,sum(value))

    
    # print(moves1==moves2)
    # print(probs1==probs2)
    # prob_eq = probs1==probs2
    # print(probs1[~prob_eq],probs2[~prob_eq])
    # print(mcts.root.visits)
    # print(mcts.Nsa[get_unique_hash(game.graph)])

if __name__=="__main__":
    # test_mcts()
    # test_batched_correctness()
    test_batched_speed()
