from GN0.alpha_zero.MCTS import MCTS, run_many_mcts
import os
from graph_game.graph_tools_games import get_graph_only_hex_game, Hex_game
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
from GN0.alpha_zero.visualize_training_data import visualize_data
from types import SimpleNamespace as sn
import wandb
import copy
from rich import print
from alive_progress import alive_bar

class Trainer():
    def __init__(self, nnet_creation_func:Callable, args, device):
        self.nnet:NNetWrapper = nnet_creation_func()
        self.nnet.batch_size = args.training_batch_size
        self.best_net:NNetWrapper = nnet_creation_func()
        self.best_net.nnet.load_state_dict(self.nnet.nnet.state_dict())
        self.best_net_player = None
        self.args = args
        self.mcts = MCTS(Hex_game(self.args.hex_size), nn=self.best_net.predict_for_mcts,args=self.args)
        self.maker_buffer = ReplayBuffer(burnin=0,capacity=self.args.capacity,device=device)
        self.breaker_buffer = ReplayBuffer(burnin=0,capacity=self.args.capacity,device=device)
        self.device = device
        self.elo = Elo(nnet_creation_function=nnet_creation_func,device=self.device)
        self.add_baselines()
        wandb.init(project="alpha_zero_hex",mode="online" if args.online_wandb else 'offline', anonymous='allow',config=copy.deepcopy(vars(self.args)))
        wandb.watch(self.nnet.nnet,log="all")
        print('[blue bold]\nconfig:', sn(**vars(self.args)))

    def add_baselines(self):
        self.elo.add_baseline(random_baseline,"random",0)
        stuff = torch.load(self.args.baseline_network_path,map_location=self.device)
        nnet = get_pre_defined("two_headed",args=stuff["args"]).to(self.device)
        nnet.load_state_dict(stuff["state_dict"])
        func = baseline_from_advantage_network(nnet,self.device)
        self.elo.add_baseline(func,"old_model",3000)

    def batch_execute_episodes(self,num_episodes):
        # game = get_graph_only_hex_game(self.args.hex_size)
        self.best_net.nnet.eval()
        episodes_left = num_episodes
        games = [Hex_game(self.args.hex_size) for _ in range(min(self.args.mcts_batch_size,num_episodes))]
        for game in games[:int(len(games)//2)]:
            game.view.gp["m"] = False
        episodes_left-=len(games)
        multi_mcts = [MCTS(game.copy(withboard=False),nn=None,args=self.args) for game in games]
        maker_train_examples = [[] for _ in range(len(games))]
        breaker_train_examples = [[] for _ in range(len(games))]

        step = 0
        win_stats = []
        with alive_bar(total=num_episodes) as bar:
            while len(games)>0:
                step += 1
                run_many_mcts(multi_mcts,self.best_net.predict_many_for_mcts,self.args.num_iterations)

                # These do not include terminal nodes
                action_temp = np.inf if step==1 else int(step < self.args.temp_threshold)
                moves,training_pi = zip(*[mcts.extract_result(mcts.rootgraph,self.args.training_temp) for mcts in multi_mcts])
                moves,action_pi = zip(*[mcts.extract_result(mcts.rootgraph,action_temp) for mcts in multi_mcts])
                
                datas = [convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])]) for game in games]
                del_ids = []
                for i in range(len(games)):
                    if games[i].view.gp["m"]:
                        maker_train_examples[i].append([datas[i],training_pi[i],None])
                    else:
                        breaker_train_examples[i].append([datas[i],training_pi[i],None])
                    action = np.random.choice(moves[i],p=action_pi[i])
                    games[i].make_move(action,remove_dead_and_captured=True)
                    win = games[i].who_won()
                    if win is not None:
                        bar()
                        for e in maker_train_examples[i]:
                            e[2] = int(win=="m")
                            self.maker_buffer.put(*e)
                        for e in breaker_train_examples[i]:
                            e[2] = int(win=="b")
                            self.breaker_buffer.put(*e)
                        maker_train_examples[i] = []
                        breaker_train_examples[i] = []
                        win_stats.append(win)
                        if episodes_left>0:
                            episodes_left-=1
                            games[i] = Hex_game(self.args.hex_size)
                            multi_mcts[i].rootgraph = Graph(games[i].graph)
                            if episodes_left%2==0:
                                games[i].view.gp["m"] = False
                        else:
                            del_ids.append(i)

                    else:
                        multi_mcts[i].rootgraph = Graph(games[i].graph)

                for i in reversed(del_ids):
                    del games[i]
                    del multi_mcts[i]
                    del maker_train_examples[i]
                    del breaker_train_examples[i]
            return win_stats

    def execute_episode(self,starting_player="m"):
        """One episode of self play"""
        # game = get_graph_only_hex_game(self.args.hex_size)
        self.best_net.nnet.eval()
        game = Hex_game(self.args.hex_size)
        if starting_player=="b":
            game.view.gp["m"] = False
        maker_train_examples = []
        breaker_train_examples = []

        step = 0
        self.mcts.rootgraph = Graph(game.graph)
        while True:
            step += 1
            self.mcts.run(self.mcts.rootgraph,self.args.num_iterations)

            # This might be different from alpha-zero, but to me it does not make
            # any sense to include varying temperatures in the training examples.
            training_temp = self.args.training_temp
            action_temp = np.inf if step==1 else int(step < self.args.temp_threshold)

            # These do not include terminal nodes
            moves,training_pi = self.mcts.extract_result(training_temp)
            moves,action_pi = self.mcts.extract_result(action_temp)
            
            data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])])
            # Account for terminal nodes:
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
                return win
            else:
                self.mcts.rootgraph = Graph(game.graph)

    def learn(self):
        log = dict()
        for epoch in range(1, self.args.num_epochs + 1):
            if self.args.batched_mcts:
                win_stats = self.batch_execute_episodes(self.args.num_episodes)
                log["winrate/maker"] = win_stats.count("m")/len(win_stats)
            else:
                maker_wins = 0
                breaker_wins = 0
                for i in trange(self.args.num_episodes):
                    winner = self.execute_episode(starting_player="m" if i%2==0 else "b")
                    if winner=="m":
                        maker_wins+=1
                    else:
                        breaker_wins+=1
                log["winrate/maker"] = maker_wins/(maker_wins+breaker_wins)
            # visualize_data(self.maker_buffer)

            log["loss/pi"],log["loss/v"],log["loss/total"] = self.nnet.train(self.maker_buffer,self.breaker_buffer,num_epochs=self.args.num_training_epochs)
            
            if "best_player" not in self.elo.players:
                prev_version_beaten = True
            else:
                self.nnet.save_checkpoint(path=os.path.join(self.args.checkpoint,"tmp_net.pt"),args=self.args)
                self.elo.add_player(os.path.join(self.args.checkpoint,"tmp_net.pt"),name="tmp_player")
                print("playing against old model")
                stats = self.elo.play_all_starting_positions("tmp_player","best_player",progress=True,hex_size=self.args.hex_size)
                winrate = stats["tmp_player"]/sum(stats.values())
                prev_version_beaten = winrate>self.args.required_beat_old_model_winrate
                log["winrate/prev_best"] = winrate

            if prev_version_beaten:
                print("New best model")
                save_path = os.path.join(self.args.checkpoint,f"{epoch}.pt")
                self.nnet.save_checkpoint(save_path)
                self.best_net.load_checkpoint(save_path)
                self.elo.add_player(save_path,"best_player")
                baseline_stats = self.elo.eval_against_baselines("best_player",hex_size=self.args.hex_size)
                for stats in baseline_stats:
                    other_player = [x for x in stats.keys() if x!="best_player"][0]
                    winrate = stats["best_player"]/sum(stats.values())
                    log[f"winrate/{other_player}"] = winrate
            else:
                print(f"failed to beat best version, winrate {winrate:.4f}")

            log["buffer_size/maker"] = len(self.maker_buffer)
            log["buffer_size/breaker"] = len(self.breaker_buffer)
            log["epoch"] = epoch
            log["previous_version_beaten"] = int(prev_version_beaten)
            print(log)
            wandb.log(log)
