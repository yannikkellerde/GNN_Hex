from GN0.alpha_zero.NN_interface import NNetWrapper
from graph_game.graph_tools_games import Hex_game,get_graph_only_hex_game
from alive_progress import alive_bar
import torch
from typing import List
from graph_game.shannon_node_switching_game import Node_switching_game
from GN0.util.convert_graph import convert_node_switching_game
from torch_geometric import Batch
import numpy as np

class Elo():
    def __init__(self,nnet_creation_function,device="cpu",k=3):
        self.players = dict()
        self.device = device
        self.K = k
        self.empty_net1:NNetWrapper = nnet_creation_function()
        self.empty_net2:NNetWrapper = nnet_creation_function()
        self.empty_net1.nnet.eval()
        self.empty_net2.nnet.eval()
        self.baselines = dict()

    def add_baseline(self,choose_move_func,name,rating):
        self.baselines[name] = {"func":choose_move_func,"rating":rating,"name":name}

    def add_player(self,checkpoint,name,initial_rating=500):
        self.players[name] = dict(name=name,checkpoint=checkpoint,rating=initial_rating,func=None)

    def eval_against_baselines(self,name,hex_size=11):
        all_stats = []
        for baseline in self.baselines:
            stats = self.play_all_starting_positions(name,baseline,hex_size=hex_size,progress=True)
            all_stats.append(stats)
        return all_stats

    def get_rating_table(self):
        columns = ["name","rating"]
        data = []
        for name,contestant in self.players.items():
            data.append([name,contestant["rating"]])
        data.sort(key=lambda x:-x[1])
        return columns,data

    def play_all_starting_positions(self,p1_name,p2_name,progress=False,hex_size=11):
        print(f"Playing games between {p1_name} and {p2_name}")
        if p1_name in self.players:
            p1 = self.players[p1_name]
        else:
            p1 = self.baselines[p1_name]
        if p2_name in self.players:
            p2 = self.players[p2_name]
        else:
            p2 = self.baselines[p2_name]
        wins = {p1_name:0,p2_name:0}
        if "checkpoint" in p1:
            self.empty_net1.load_checkpoint(p1["checkpoint"])
            p1["func"] = self.empty_net1.choose_moves

        if "checkpoint" in p2:
            self.empty_net2.load_checkpoint(p2["checkpoint"])
            p2["func"] = self.empty_net2.choose_moves
        
        some_game = Hex_game(hex_size)
        starting_moves = some_game.get_unique_starting_moves()
        with alive_bar(len(starting_moves)*4,disable=not progress) as bar:
            with torch.no_grad():
                for i in range(4):
                    games = [get_graph_only_hex_game(hex_size) for _ in range(len(starting_moves))]
                    if i>=2:
                        for game in games:
                            game.view.gp["m"] = False
                    if i%2==0:
                        current_player = p1
                    else:
                        current_player = p2
                    
                    move_num = 0
                    while len(games)>0:
                        if move_num == 0:
                            actions = starting_moves
                        else:
                            actions = current_player["func"](games)
                        to_del = []

                        for i,action in enumerate(actions):
                            games[i].make_move(action,remove_dead_and_captured=True)
                            winner = games[i].who_won()

                            if winner is not None:
                                bar()
                                if winner == games[i].not_onturn:
                                    wins[current_player["name"]]+=1
                                else:
                                    wins[p1 if current_player==p2 else p1] += 1
                                to_del.append(i)
                        for i in reversed(to_del):
                            del games[i]
                        current_player = p1 if current_player==p2 else p2
                        move_num += 1
        return wins

def baseline_from_advantage_network(nnet,device):
    def choose_moves(games:List[Node_switching_game]):
        datas = [convert_node_switching_game(game.view,global_input_properties=[game.view.gp["m"]]) for game in games]
        batch = Batch.from_data_list(datas)
        action_values = nnet.simple_forward(batch.to(device)).to(device)
        actions = []
        for (start,fin) in zip(batch.ptr,batch.ptr[1:]):
            action_part = action_values[start+2:fin]
            if len(action_part) == 1:
                action = 2
            else:
                action = torch.argmax(action_part)+2
            actions.append(action)
        actions = [datas[i].backmap[actions[i]].item() for i in range(len(actions))]
        return actions

def random_baseline(games:List[Node_switching_game]):
    return [np.random.choice(game.get_actions()) for game in games]
