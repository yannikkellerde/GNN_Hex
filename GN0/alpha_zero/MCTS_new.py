from graph_game.shannon_node_switching_game import Node_switching_game
from graph_tool.all import Graph,GraphView
import numpy as np
from typing import Callable
from graph_game.graph_tools_hashing import get_unique_hash
from GN0.util.util import get_one_hot
from time import perf_counter
from collections import defaultdict
from typing import List
from tqdm import trange

class MCTS():
    def __init__(self,game:Node_switching_game,nn:Callable,args,remove_dead_and_captured=True,debug=False):
        self.game = game
        self.rootgraph = Graph(self.game.graph)
        self.nn = nn
        self.cpuct = args.cpuct

        self.Qsa = {}  # stores all Q values for s as a np array
        self.Nsa = {}  # stores #times edges s,a where visited, as np array
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.mode = None # If a leaf or a terminal was found
        self.path = [] # The most recent path
        self.v = 0
        self.leaf_graph = None
        self.remove_dead_and_captured = remove_dead_and_captured
        self.debug = debug
        self.timers = defaultdict(list)

    def extract_result(self,graph:Graph,temp=1):
        assert not isinstance(graph,GraphView) # Call with graph, not with view!
        self.game.set_to_graph(graph)
        s = get_unique_hash(self.game.view)
        
        if temp == 0:
            bestAs = np.array(np.argwhere(self.Nsa[s]==np.max(self.Nsa[s]))).flatten()
            bestA = np.random.choice(bestAs)
            probs = get_one_hot(len(self.Nsa[s]),bestA)
        elif temp == np.inf:
            probs = np.ones(len(self.Nsa[s]))/len(self.Nsa[s])
        else:
            counts = self.Nsa[s]**(1./temp)
            s = np.sum(counts)
            if s==0:
                probs = np.ones(len(self.Nsa[s]))/len(self.Nsa[s])
            else:
                probs = counts/s
        return self.game.get_actions(),probs

    def process_results(self,value=None,pi=None):
        start = perf_counter()
        if self.mode == 'leaf':
            assert pi is not None and value is not None
            s = self.path.pop()[0]
            self.Ps[s] = pi.cpu().numpy()
            self.Ns[s] = 1
            self.v = 1-value

        for s,a in reversed(self.path):
            if s in self.Qsa:
                self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + self.v) / (self.Nsa[s][a] + 1)
                self.Nsa[s][a] += 1
            else:
                self.Qsa[s] = np.ones(len(self.Ps[s]))*0.5
                self.Nsa[s] = np.zeros(len(self.Ps[s]))
                self.Qsa[s][a] = self.v
                self.Nsa[s][a] = 1
            self.Ns[s] += 1
            self.v = 1-self.v
        self.path = []
        self.timers["process_results"].append(perf_counter()-start)

    def find_leaf(self,set_to_graph=None):
        """This is for multi-mcts (batched processing)"""
        assert not isinstance(set_to_graph,GraphView) # Call with graph, not with view!
        if set_to_graph is not None:
            self.game.set_to_graph(set_to_graph)

        ###########################
        start = perf_counter()
        ###########################
        s = get_unique_hash(self.game.view)
        ###########################
        self.timers["hash"].append(perf_counter()-start)
        ###########################

        if s not in self.Es:
            self.Es[s] = self.game.who_won()

        if self.Es[s] is not None:
            # terminal node
            self.mode = 'terminal'
            self.v = 0 if self.Es[s]==self.game.onturn else 1
            return False

        if s not in self.Ps:
            self.mode = 'leaf'
            self.path.append((s,None))
            return True

        actions = self.game.get_actions()
        assert len(actions)==len(self.Ps[s])
        if s in self.Qsa:
            # compute upper confidence bound
            u = self.Qsa[s] + self.cpuct * self.Ps[s] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[s])
        else:
            # Assume Q = 0.5 for unknown nodes
            u = 0.5 + self.cpuct * self.Ps[s] * np.sqrt(self.Ns[s])

        a = np.argmax(u)
        action = actions[a]
        start = perf_counter()
        self.game.make_move(action,remove_dead_and_captured=self.remove_dead_and_captured)
        self.timers["make_move"].append(perf_counter()-start)
        self.path.append((s,a))
        return self.find_leaf()


    def run(self,num_iterations,graph):
        """This is for single mcts processing (No batching)"""
        for _ in range(num_iterations):
            self.single_iteration(Graph(graph))

    def single_iteration(self,set_to_graph=None):
        """This is for single mcts processing (No batching)"""
        ###########################
        start = perf_counter()
        ###########################

        assert not isinstance(set_to_graph,GraphView) # Call with graph, not with view!
        if set_to_graph is not None:
            self.game.set_to_graph(set_to_graph)

        ###########################
        self.timers["set to graph"].append(perf_counter()-start)
        start = perf_counter()
        ###########################

        s = get_unique_hash(self.game.view)

        ###########################
        self.timers["hashing"].append(perf_counter()-start)
        ###########################

        if s not in self.Es:
            self.Es[s] = self.game.who_won()
        if self.Es[s] is not None:
            # terminal node
            if self.debug:
                self.leaf_graph = Graph(self.game.graph) # Useful for debugging and visualization
            return 0 if self.Es[s]==self.game.onturn else 1

        if s not in self.Ps:
            ###########################
            start = perf_counter()
            ###########################
            # leaf node
            prior_tensor, v = self.nn(self.game)
            self.Ps[s] = prior_tensor.cpu().numpy()
            self.Ns[s] = 1  # Reference implementations say so, but I don't think nodes should start with 0 visits
            if self.debug:
                self.leaf_graph = Graph(self.game.graph)  # Useful for debugging and visualization
            ###########################
            self.timers["nn"].append(perf_counter()-start)
            ###########################
            return 1-v

        ###########################
        start = perf_counter()
        ###########################

        actions = self.game.get_actions()
        if s in self.Qsa:
            # compute upper confidence bound
            u = self.Qsa[s] + self.cpuct * self.Ps[s] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[s])
        else:
            # Assume Q = 0.5 for unknown nodes
            u = 0.5 + self.cpuct * self.Ps[s] * np.sqrt(self.Ns[s])

        a = np.argmax(u)
        self.game.make_move(actions[a],remove_dead_and_captured=self.remove_dead_and_captured)
        ###########################
        self.timers["moving"].append(perf_counter()-start)
        ###########################

        v = self.single_iteration()

        ###########################
        start = perf_counter()
        ###########################
        if s in self.Qsa:
            self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + v) / (self.Nsa[s][a] + 1)
            self.Nsa[s][a] += 1
        else:
            self.Qsa[s] = np.ones(len(actions))*0.5
            self.Nsa[s] = np.zeros(len(actions))
            self.Qsa[s][a] = v
            self.Nsa[s][a] = 1

        self.Ns[s] += 1
        ###########################
        self.timers["rest"].append(perf_counter()-start)
        ###########################
        return 1-v

def run_many_mcts(many_mcts:List[MCTS],nn:Callable,num_iterations:int,progress=False):
    for i in (trange(num_iterations) if progress else range(num_iterations)):
        mcts_maker = []
        mcts_breaker = []
        for mcts in many_mcts:
            need_nn = mcts.find_leaf(set_to_graph=Graph(mcts.rootgraph))
            if need_nn:
                if mcts.game.view.gp["m"]:
                    mcts_maker.append(mcts)
                else:
                    mcts_breaker.append(mcts)
            else:
                mcts.process_results()
        for some_mcts in (mcts_maker,mcts_breaker):
            if len(some_mcts)>0:
                nn_output = nn([mcts.game for mcts in some_mcts])
                for mcts,(pi,value) in zip(some_mcts,nn_output):
                    mcts.process_results(pi=pi,value=value)
