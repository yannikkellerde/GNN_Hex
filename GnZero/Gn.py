import graph_nets as gn
import sonnet as snt
import tensorflow as tf
from functools import reduce
import logging
from GnZero.convert_graph import convert_graph
from GnZero.MCTS import MCTS
from GnZero.models import EncodeProcessDecode
from game.graph_tools_game import Graph_game

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
            "layers":2
        },
        "node_model": {
            "latent_size":16,
            "layers":2
        },
        "global_model": {
            "latent_size":16,
            "layers":2
        }
    },
    "core_settings": {
        "edge_model": {
            "latent_size":16,
            "layers":2
        },
        "node_model": {
            "latent_size":16,
            "layers":2
        },
        "global_model": {
            "latent_size":16,
            "layers":2
        }
    },
    "decoder_settings": {
        "edge_model": {
            "latent_size":16,
            "layers":2
        },
        "node_model": {
            "latent_size":16,
            "layers":2
        },
        "global_model": {
            "latent_size":16,
            "layers":2
        }
    },
    "edge_fn":None,
    "node_fn":lambda:snt.nets.MLP([1],activation=tf.keras.activations.tanh,activate_final=True),
    "global_fn":lambda:snt.nets.MLP([1],activation=tf.keras.activations.tanh,activate_final=True)
}

class Gn():
    def __init__(self,game:Graph_game):
        self.game = game
        self.root_storage = self.game.extract_storage()
        self.model = EncodeProcessDecode(**model_architecture)
        self.lr = 1e-3
        self.optimizer = snt.optimizers.Adam(learning_rate)
        self.self_play_iterations = 20
        self.temperature = 1
    
    def self_play(self,num_games=10):
        training_examples = {
            "inputs":[],
            "target_probs":[],
            "target_values":[]
        }
        for n in num_games:
            inputs = []
            target_probs = []
            self.game.load_storage(self.root_storage)
            mcts = MCTS(self.game)
            done = False
            while not done:
                mcts.run(self.self_play_iterations)
                moves,probs = mcts.extract_result(self.temperature)
                self.game.load_storage(mcts.root.storage)
                inputs.append(convert_graph(self.game.view))
                