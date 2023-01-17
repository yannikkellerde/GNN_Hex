import os
import numpy as np
import random
from graph_game.graph_tools_games import Hex_game
from GN0.util.convert_graph import convert_node_switching_game
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from alive_progress import alive_bar
from GN0.models import get_pre_defined
from collections import defaultdict,deque
from collections import defaultdict
from argparse import Namespace
import matplotlib.pyplot as plt
import json
import wandb


class Elo_handler():
    def __init__(self,hex_size,empty_model_func=None,device="cpu",k=3):
        self.players = {}
        self.size = hex_size
        self.elo_league_contestants = list()
        self.device = device
        self.K = k
        if empty_model_func is not None:
            self.create_empty_models(empty_model_func)

    def reset(self,new_hex_size=None,keep_players=[]):
        new_players = {}
        for keep_player in keep_players:
            if keep_player in self.players:
                new_players[keep_player] = self.players[keep_player]
        self.size = new_hex_size
        self.players = new_players

    
    def create_empty_models(self,empty_model_func):
        self.empty_model1 = empty_model_func().to(self.device)
        self.empty_model1.eval()
        self.empty_model2 = empty_model_func().to(self.device)
        self.empty_model2.eval()

    def add_player(self,name,model=None,set_rating=None,simple=False,rating_fixed=False,episode_number=None,checkpoint=None):
        self.players[name] = {"model":model,"simple":simple,"rating":set_rating,"rating_fixed":rating_fixed,"episode_number":episode_number,"checkpoint":checkpoint}

    def roundrobin(self,num_players,num_games_per_match,must_include_players=[]):
        assert num_players>len(must_include_players)
        ok_players = [x for x in self.players if self.players[x]["rating"] is not None and not self.players[x]["rating_fixed"]]
        if num_players > len(ok_players):
            num_players = len(ok_players)
        contestants = []
        contestants.extend(must_include_players)
        while len(contestants)<num_players:
            new_name = random.choice(ok_players)
            if new_name not in contestants:
                contestants.append(new_name)

        all_stats = []

        with alive_bar(len(contestants)*(len(contestants)-1),disable=True) as bar:
            for p1 in contestants:
                for p2 in contestants:
                    if p1==p2:
                        continue
                    if not "model" in self.players[p1] or self.players[p1]["model"] is None:
                        self.players[p1]["model"] = self.empty_model1
                    if "checkpoint" in self.players[p1] and self.players[p1]["checkpoint"] is not None:
                        self.load_into_empty_model(self.players[p1]["model"],self.players[p1]["checkpoint"])

                    if not "model" in self.players[p2] or self.players[p2]["model"] is None:
                        self.players[p2]["model"] = self.empty_model2
                    if "checkpoint" in self.players[p2] and self.players[p2]["checkpoint"] is not None:
                        self.load_into_empty_model(self.players[p2]["model"],self.players[p2]["checkpoint"])
                    statistics = self.play_some_games(p1,p2,num_games_per_match,0,random_first_move=False,progress=False)
                    all_stats.append(statistics)
                    bar()

        self.score_some_statistics(all_stats)

    def load_into_empty_model(self,empty_model,checkpoint):
        stuff = torch.load(checkpoint,map_location=self.device)
        empty_model.load_state_dict(stuff["state_dict"])
        if "cache" in stuff and stuff["cache"] is not None:
            empty_model.import_norm_cache(*stuff["cache"])

    def score_some_statistics(self,statistics):
        player_expect_vs_score = defaultdict(lambda :dict(expectation=0,score=0,num_games=0))
        numerator_and_games = defaultdict(lambda: dict(numerator=0,num_games=0)) # for initial rating

        for stats in statistics:
            keys = list(stats.keys())
            num_games = sum([int(x) for x in stats.values()])
            if self.get_rating(keys[0]) is None:
                if self.get_rating(keys[1]) is not None:
                    numerator_and_games[keys[0]]["numerator"] += self.get_rating(keys[1])*num_games+400*(stats[keys[0]]*2-num_games)
                    numerator_and_games[keys[0]]["num_games"] += num_games
            elif self.get_rating(keys[1]) is not None:
                player_expect_vs_score[keys[0]]["expectation"] += (1/(1+10**((self.get_rating(keys[1])-self.get_rating(keys[0]))/400)))*num_games
                player_expect_vs_score[keys[0]]["score"] += stats[keys[0]]
            if self.get_rating(keys[1]) is None:
                if self.get_rating(keys[0]) is not None:
                    numerator_and_games[keys[1]]["numerator"] += self.get_rating(keys[0])*num_games+400*(stats[keys[1]]*2-num_games)
                    numerator_and_games[keys[1]]["num_games"] += num_games
            elif self.get_rating(keys[0]) is not None:
                player_expect_vs_score[keys[1]]["expectation"] += (1/(1+10**((self.get_rating(keys[0])-self.get_rating(keys[1]))/400)))*num_games
                player_expect_vs_score[keys[1]]["score"] += stats[keys[1]]
        
        for key in player_expect_vs_score:
            if not self.players[key]["rating_fixed"]:
                self.players[key]["rating"] += self.K*(player_expect_vs_score[key]["score"]-player_expect_vs_score[key]["expectation"])

        for key in numerator_and_games:
            if not self.players[key]["rating_fixed"]:
                self.players[key]["rating"] = numerator_and_games[key]["numerator"]/numerator_and_games[key]["num_games"]


    def get_rating_table(self):
        columns = ["name","rating"]
        data = []
        for name in self.players:
            if self.get_rating(name) is not None:
                data.append([name,self.get_rating(name)])
        data.sort(key=lambda x:-x[1])
        return columns,data

    def plt_elo(self):
        datapairs = []
        for name in self.players:
            if self.players[name]["episode_number"] is not None and self.players[name]["rating"] is not None:
                datapairs.append((self.players[name]["episode_number"],self.players[name]["rating"]))
        table = wandb.Table(data=datapairs, columns = ["game_frame", "elo"])
        return wandb.plot.line(table,"game_frame","elo")

    def get_rating(self,player_name):
        return self.players[player_name]["rating"]

    def play_some_games(self,maker,breaker,num_games,temperature,random_first_move=False,progress=False):
        print("Playing games between",maker,"and",breaker)
        wins = {maker:0,breaker:0}
        game_lengths = []
        some_game = Hex_game(self.size)
        starting_moves = some_game.get_unique_starting_moves()
        random.shuffle(starting_moves)

        if num_games is None:
            num_games = len(starting_moves)*2
        if num_games > len(starting_moves)*2:
            num_games = len(starting_moves)*2

        with alive_bar(num_games,disable=not progress) as bar:
            with torch.no_grad():
                for i in range(2):
                    if i==0:
                        p1 = maker
                        p2 = breaker
                    else:
                        p1 = breaker
                        p2 = maker

                    games = [Hex_game(self.size) for _ in range(num_games//2)]
                    if p1 == breaker:
                        for game in games:
                            game.view.gp["m"] = False
                    for game in games:
                        game.board_callback = game.board.graph_callback
                    move_num = 0
                    current_player = p1

                    while len(games)>0:
                        datas = [convert_node_switching_game(game.view,global_input_properties=[game.view.gp["m"]], need_backmap=True,old_style=True) for game in games]
                        batch = Batch.from_data_list(datas)

                        if move_num == 0 and random_first_move and starting_moves is None:
                            actions = [random.choice(g.get_actions()) for g in games]
                        elif move_num>0 or starting_moves is None:
                            if self.players[current_player]["simple"]:
                                actions = self.players[current_player]["model"](batch)
                            else:
                                action_values = self.players[current_player]["model"].simple_forward(batch.to(self.device)).to(self.device)
                                actions = []
                                for i,(start,fin) in enumerate(zip(batch.ptr,batch.ptr[1:])):  # This isn't great, but I didn't find any method for sampling in pytorch_scatter. Maybe need to implement myself at some point.
                                    action_part = action_values[start+2:fin]
                                    if len(action_part)==1:
                                        actions.append(2)
                                        continue
                                    if temperature==0:
                                        try:
                                            action = torch.argmax(action_part)+2
                                        except Exception as e:
                                            print(action_part,type(action_part))
                                            print(action_part.size())
                                            raise ValueError(str(e))
                                        actions.append(action)
                                        continue
                                    prob_part = F.softmax(action_part/temperature, dim=0)
                                    try:
                                        distrib = Categorical(prob_part.squeeze())
                                    except ValueError:
                                        raise ValueError
                                    sample = distrib.sample()
                                    action = sample+2
                                    actions.append(action.item())
                            actions = [datas[i].backmap[actions[i]].item() for i in range(len(actions))]
                        elif move_num==0 and starting_moves is not None:
                            actions = starting_moves[:len(games)]


                        to_del = []
                        # if actions[0]!=0:
                        #     print(games[0].board.vertex_to_board_index[games[0].view.vertex(actions[0])])
                        #     print(f"Red:{p1}, Blue:{p2}, onturn: {games[0].onturn}, action: {games[0].board.vertex_index_to_string_move(actions[0])}")
                        #     print(games[0].board.draw_me())
                        for i,action in enumerate(actions):
                            if action!=0:
                                games[i].make_move(action,remove_dead_and_captured=True)
                                winner = games[i].who_won()
                                
                                if winner is not None:
                                    bar()
                                    game_lengths.append(games[i].total_num_moves)
                                    if winner == games[i].not_onturn:
                                        wins[current_player]+=1
                                    else:
                                        wins[p1 if current_player==p2 else p1] += 1
                                    to_del.append(i)
                        for i in reversed(to_del):
                            del games[i]

                        if current_player == p1:
                            current_player = p2
                        else:
                            current_player = p1
                        move_num += 1

            print("scoring games")
            print("Mean game length",np.mean(game_lengths))

            statistics = wins.copy()
            print("stats:",statistics)
            # to_score_games = []
            # while wins[maker]>0 or wins[breaker]>0:
            #     if wins[maker]>0:
            #         to_score_games.append({"winner":maker,"loser":breaker,"winnerHome":True})
            #         wins[maker]-=1
            #     if wins[breaker]>0:
            #         to_score_games.append({"winner":breaker,"loser":maker,"winnerHome":True})
            #         wins[breaker]-=1
            return statistics


def random_player(batch):
    # print(batch.ptr[1:]-batch.ptr[:-1])

    actions = [random.randint(2,after-before-1) for before,after in zip(batch.ptr,batch.ptr[1:])]
    return actions

def evaluate_checkpoint_against_random_mover(elo_handler:Elo_handler, checkpoint, model):
    stuff = torch.load(checkpoint)
    model.load_state_dict(stuff["state_dict"])
    model.import_norm_cache(*stuff["cache"])
    model.eval()
    model.to(device)
    elo_handler.add_player("model",model)
    elo_handler.add_player("random",random_player,set_rating=1500,simple=True)
    res = elo_handler.play_some_games("model","random",num_games=64,temperature=0,progress=True)
    # res = elo_handler.play_some_games("model","model2",num_games=64,temperature=0.0001,random_first_move=False)
    res = elo_handler.play_some_games("random","model",num_games=64,temperature=0,progress=True)
    print(res)
    print(elo_handler.get_rating("model"))
    print(elo_handler.get_rating("random"))


if __name__ == "__main__":
    from Rainbow.common.utils import get_highest_model_path
    # elo_handler = Elo_handler(9)
    # checkpoint = "Rainbow/checkpoints/worldly-fire-19/checkpoint_4499712.pt"
    # model = get_pre_defined("sage+norm")
    # evaluate_checkpoint_against_random_mover(elo_handler,checkpoint,model)
    # run_league("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/ethereal-glitter-22")
    # test_elo_handler()
    # battle_it_out()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # battle_it_out(device=device)
    # old_vs_new(old_breaker_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_breaker_32800000.pt",old_maker_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_maker_32800000.pt",old_model_name="sage+norm",new_model_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/azure-snowball-157/checkpoint_59200000.pt",new_model_name="two_headed")
