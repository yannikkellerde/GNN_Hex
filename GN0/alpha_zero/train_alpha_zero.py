from GN0.alpha_zero.MCTS import MCTS
from graph_game.graph_tools_games import get_graph_only_hex_game, Hex_game
from graph_game.shannon_node_switching_game import Storage
from graph_tool.all import Graph
from GN0.util.convert_graph import convert_node_switching_game
import numpy as np
from GN0.alpha_zero.replay_buffer import ReplayBuffer
from tqdm import trange
from GN0.alpha_zero.NN_interface import NNetWrapper

class Trainer():
    def __init__(self, nnet:NNetWrapper, args, device):
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(Hex_game(self.args.hex_size), self.nnet.predict_for_mcts)
        self.maker_buffer = ReplayBuffer(burnin=0,capacity=self.args.capacity,device=device)
        self.breaker_buffer = ReplayBuffer(burnin=0,capacity=self.args.capacity,device=device)
        self.device = device

    def execute_episode(self):
        """One episode of self play"""
        # game = get_graph_only_hex_game(self.args.hex_size)
        self.nnet.nnet.eval()
        game = Hex_game(self.args.hex_size)
        maker_train_examples = []
        breaker_train_examples = []

        step = 0
        self.mcts.reset(storage=Graph(game.graph))
        while True:
            step += 1
            self.mcts.run(self.args.num_iterations)

            # This might be different from alpha-zero, but to me it does not make
            # any sense to include varying temperatures in the training examples.
            training_temp = self.args.training_temp
            action_temp = np.inf if step==1 else int(step < self.args.temp_threshold)

            moves,training_pi = self.mcts.extract_result(training_temp)
            moves,action_pi = self.mcts.extract_result(action_temp)
            
            data = convert_node_switching_game(game.view,global_input_properties=int(game.view.gp["m"]))
            if game.view.gp["m"]:
                maker_train_examples.append([data,training_pi,None])
            else:
                breaker_train_examples.append([data,training_pi,None])

            action = np.random.choice(moves,p=action_pi)
            game.make_move(action,remove_dead_and_captured=True)
            win = game.who_won()
            if win is not None:
                for e in maker_train_examples:
                    e[2] = int(win=="m")
                    self.maker_buffer.put(*e)
                for e in breaker_train_examples:
                    e[2] = int(win=="b")
                    self.breaker_buffer.put(*e)
                return
            else:
                self.mcts.next_iter_with_child(action,Graph(game.graph))

    def learn(self):
        for epoch in range(1, self.args.num_epochs + 1):
            for _ in trange(self.args.num_episodes):
                self.execute_episode()




