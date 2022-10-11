from GN0.alpha_zero.MCTS import MCTS
import os
from graph_game.graph_tools_games import get_graph_only_hex_game, Hex_game
from graph_game.shannon_node_switching_game import Storage
from graph_tool.all import Graph
from GN0.util.convert_graph import convert_node_switching_game
import numpy as np
from GN0.alpha_zero.replay_buffer import ReplayBuffer
from tqdm import trange
from GN0.alpha_zero.NN_interface import NNetWrapper
from typing import Callable
from GN0.alpha_zero.elo import Elo, random_baseline, baseline_from_advantage_network
import torch
from GN0.models import get_pre_defined
import wandb

class Trainer():
    def __init__(self, nnet_creation_func:Callable, args, device):
        self.nnet:NNetWrapper = nnet_creation_func()
        self.best_net:NNetWrapper = nnet_creation_func()
        self.best_net.nnet.load_state_dict(self.nnet.nnet.state_dict())
        self.best_net_player = None
        self.args = args
        self.mcts = MCTS(Hex_game(self.args.hex_size), self.best_net.predict_for_mcts)
        self.maker_buffer = ReplayBuffer(burnin=0,capacity=self.args.capacity,device=device)
        self.breaker_buffer = ReplayBuffer(burnin=0,capacity=self.args.capacity,device=device)
        self.device = device
        self.elo = Elo(nnet_creation_function=nnet_creation_func,device=self.device)
        self.add_baselines()
        wandb.init(project="alpha_zero_hex",mode="online" if args.online_wandb else 'offline', anonymous='allow')

    def add_baselines(self):
        self.elo.add_baseline(random_baseline,"random",0)
        stuff = torch.load(self.args.baseline_network_path)
        nnet = get_pre_defined("two_headed",args=stuff["args"])
        nnet.load_state_dict(stuff["state_dict"])
        func = baseline_from_advantage_network(nnet,self.device)
        self.elo.add_baseline(func,"old_model",3000)

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

            # These do not include terminal nodes
            moves,training_pi = self.mcts.extract_result(training_temp)
            moves,action_pi = self.mcts.extract_result(action_temp)
            
            data = convert_node_switching_game(game.view,global_input_properties=int(game.view.gp["m"]))
            # Account for terminal nodes:
            data_pi = np.zeros(len(training_pi)+2)
            data_pi[2:] = training_pi
            if game.view.gp["m"]:
                maker_train_examples.append([data,data_pi,None])
            else:
                breaker_train_examples.append([data,data_pi,None])

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

            self.nnet.train(self.maker_buffer,self.breaker_buffer,num_epochs=self.args.num_training_epochs)
            
            if "best_player" not in self.elo.players:
                prev_version_beaten = True
            else:
                self.nnet.save_checkpoint(path=os.path.join(self.args.checkpoint,"tmp_net.pt"))
                self.elo.add_player(os.path.join(self.args.checkpoint,"tmp_net.pt"),name="tmp_player")
                print("playing against old model")
                stats = self.elo.play_all_starting_positions(self.elo.players["tmp_player"],self.elo.players["best_player"],progress=True,hex_size=self.args.hex_size)
                winrate = stats["tmp_player"]/sum(stats.values())
                prev_version_beaten = winrate>self.args.required_beat_old_model_winrate

            if prev_version_beaten:
                print("New best model")
                save_path = os.path.join(self.args.checkpoint,f"{epoch}.pt")
                self.nnet.save_checkpoint(save_path)
                self.best_net.load_checkpoint(save_path)
                self.elo.add_player(save_path,"best_player")
                baseline_stats = self.elo.eval_against_baselines("best_player",hex_size=self.args.hex_size)
            else:
                print(f"failed to beat best version, winrate {winrate:.4f}")

