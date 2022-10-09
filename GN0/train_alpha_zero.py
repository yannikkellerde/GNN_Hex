from GN0.MCTS import MCTS,format_nn
from graph_game.graph_tools_games import get_graph_only_hex_game, Hex_game
from graph_game.shannon_node_switching_game import Storage
from graph_tool.all import Graph
from GN0.convert_graph import convert_node_switching_game
import numpy as np

class Trainer():
    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(Hex_game(self.args.hex_size), format_nn(self.nnet))
        self.maker_train_examples = []
        self.breaker_train_examples = []

    def execute_episode(self):
        """One episode of self play"""
        # game = get_graph_only_hex_game(self.args.hex_size)
        game = Hex_game(self.args.hex_size)

        step = 0
        while True:
            step += 1
            self.mcts.reset(storage=Graph(game.graph))
            self.mcts.run(self.args.num_iterations)

            # This might be different from alpha-zero, but to me it does not make
            # any sense to include varying temperatures in the training examples.
            training_temp = self.args.training_temp
            action_temp = np.inf if step==1 else int(step < self.args.temp_threshold)

            training_pi = self.mcts.extract_result(training_temp)
            action_pi = self.mcts.extract_result(action_temp)
            
            data = convert_node_switching_game(game.view,global_input_properties=int(game.view.gp["m"]))
            if game.view.gp["m"]:
                self.maker_train_examples


