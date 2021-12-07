import graph_nets as gn
import sonnet as snt
import tensorflow as tf
from functools import reduce
import logging
from GNZero.convert_graph import convert_graph
from GNZero.MCTS import MCTS
from GNZero.models import NN_interface
import GNZero.util as util
from game.graph_tools_game import Graph_game
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

model_architecture = {
    "encoder_settings": {
        "edge_model": {
            "latent_size":16,
            "num_layers":2
        },
        "node_model": {
            "latent_size":16,
            "num_layers":2
        },
        "global_model": {
            "latent_size":16,
            "num_layers":2
        }
    },
    "core_settings": {
        "edge_model": {
            "latent_size":16,
            "num_layers":2
        },
        "node_model": {
            "latent_size":16,
            "num_layers":2
        },
        "global_model": {
            "latent_size":16,
            "num_layers":2
        }
    },
    "decoder_settings": {
        "edge_model": {
            "latent_size":16,
            "num_layers":2
        },
        "node_model": {
            "latent_size":16,
            "num_layers":2
        },
        "global_model": {
            "latent_size":16,
            "num_layers":2
        }
    },
    "edge_fn":None,
    "node_fn":lambda:snt.nets.MLP([1],activation=tf.keras.activations.sigmoid,activate_final=True),
    "global_fn":lambda:snt.nets.MLP([1],activation=tf.keras.activations.tanh,activate_final=True)
}

class GN0():
    def __init__(self,game:Graph_game):
        self.game = game
        self.root_storage = self.game.extract_storage()
        self.model = NN_interface(model_architecture,10)
        self.lr = 1e-3
        self.optimizer = snt.optimizers.Adam(self.lr)
        self.self_play_iterations = 20
        self.temperature = 1

    def self_play(self,num_games=10):
        training_examples = {
            "inputs":[],
            "target_probs":[],
            "target_values":[]
        }
        for n in range(num_games):
            inputs = []
            target_probs = []
            self.game.load_storage(self.root_storage)
            start_onturn = self.game.onturn
            mcts = MCTS(self.game,self.model.do_policy_and_value)
            done = False
            values = None
            while not done:
                mcts.run(self.self_play_iterations)
                moves,probs = mcts.extract_result(self.temperature)
                self.game.load_storage(mcts.root.storage)
                graph,vertexmap = convert_graph(self.game.view)
                inputs.append((graph,vertexmap))
                target_probs.append((probs,moves))
                move = np.random.choice(moves,p=probs)
                done = self.game.make_move(move)
                mcts.reset(self.game.extract_storage())
                if self.game.view.num_vertices() == 0 and not done:
                    values = np.zeros(len(inputs))
                    done = True
            if values is None:
                if self.game.onturn == start_onturn:
                    values = util.get_alternating(len(inputs),-1,1)
                else:
                    values = util.get_alternating(len(inputs),1,-1)
            training_examples["inputs"].extend(inputs)
            training_examples["target_probs"].extend(target_probs)
            training_examples["target_values"].extend(values)
        return training_examples

