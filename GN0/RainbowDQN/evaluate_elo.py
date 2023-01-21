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
import pandas as pd
import matplotlib.pyplot as plt
import json
import wandb
from GN0.RainbowDQN.mohex_communicator import MohexPlayer


class Elo_handler():
    def __init__(self,hex_size,empty_model_func=None,device="cpu",k=10):
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

    def add_player(self,name,model=None,set_rating=None,simple=False,rating_fixed=False,episode_number=None,checkpoint=None,can_join_roundrobin=True,uses_empty_model=True):
        self.players[name] = {"model":model,"simple":simple,"rating":set_rating,"rating_fixed":rating_fixed,"episode_number":episode_number,"checkpoint":checkpoint,"can_join_roundrobin":can_join_roundrobin,"uses_empty_model":uses_empty_model}

    def roundrobin(self,num_players,num_games_per_match,must_include_players=[],score_as_n_games=20):
        ok_players = [x for x in self.players if self.players[x]["can_join_roundrobin"]]
        if num_players is None or num_players > len(ok_players):
            num_players = len(ok_players)
        contestants = []
        contestants.extend(must_include_players)
        while len(contestants)<num_players:
            new_name = random.choice(ok_players)
            if new_name not in contestants:
                contestants.append(new_name)

        all_stats = []

        with alive_bar(len(contestants)*(len(contestants)-1),disable=False) as bar:
            for p1 in contestants:
                for p2 in contestants:
                    if p1==p2:
                        continue
                    if self.players[p1]["uses_empty_model"]:
                        self.players[p1]["model"] = self.empty_model1
                        self.load_into_empty_model(self.players[p1]["model"],self.players[p1]["checkpoint"])

                    if self.players[p2]["uses_empty_model"]:
                        self.players[p2]["model"] = self.empty_model2
                        self.load_into_empty_model(self.players[p2]["model"],self.players[p2]["checkpoint"])
                    statistics = self.play_some_games(p1,p2,num_games_per_match,0,random_first_move=False,progress=False)
                    all_stats.append(statistics)
                    bar()

        for _ in range(score_as_n_games):
            self.score_some_statistics(all_stats)
        performances = list(tuple(self.get_performances_from_stats(all_stats).items()))
        performances.sort(key=lambda x:-x[1])
        return ["name","performance"], performances


    def load_into_empty_model(self,empty_model,checkpoint):
        stuff = torch.load(checkpoint,map_location=self.device)
        empty_model.load_state_dict(stuff["state_dict"])
        if "cache" in stuff and stuff["cache"] is not None:
            empty_model.import_norm_cache(*stuff["cache"])

    def get_performances_from_stats(self,statistics):
        performances = {}
        numerator_and_games = defaultdict(lambda: dict(numerator=0,num_games=0)) # for initial rating
        for stats in statistics:
            keys = list(stats.keys())
            num_games = sum([int(x) for x in stats.values()])
            if self.get_rating(keys[1]) is not None:
                numerator_and_games[keys[0]]["numerator"] += self.get_rating(keys[1])*num_games+400*(stats[keys[0]]*2-num_games)
                numerator_and_games[keys[0]]["num_games"] += num_games
            if self.get_rating(keys[0]) is not None:
                numerator_and_games[keys[1]]["numerator"] += self.get_rating(keys[0])*num_games+400*(stats[keys[1]]*2-num_games)
                numerator_and_games[keys[1]]["num_games"] += num_games
        for key in numerator_and_games:
            performances[key] = numerator_and_games[key]["numerator"]/numerator_and_games[key]["num_games"]
        return performances

    def score_some_statistics(self,statistics,game_num_independent=True):
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
                player_expect_vs_score[keys[0]]["num_games"] += num_games
            if self.get_rating(keys[1]) is None:
                if self.get_rating(keys[0]) is not None:
                    numerator_and_games[keys[1]]["numerator"] += self.get_rating(keys[0])*num_games+400*(stats[keys[1]]*2-num_games)
                    numerator_and_games[keys[1]]["num_games"] += num_games
            elif self.get_rating(keys[0]) is not None:
                player_expect_vs_score[keys[1]]["expectation"] += (1/(1+10**((self.get_rating(keys[0])-self.get_rating(keys[1]))/400)))*num_games
                player_expect_vs_score[keys[1]]["score"] += stats[keys[1]]
                player_expect_vs_score[keys[1]]["num_games"] += num_games
        
        for key in player_expect_vs_score:
            if not self.players[key]["rating_fixed"]:
                change = self.K*(player_expect_vs_score[key]["score"]-player_expect_vs_score[key]["expectation"])
                if game_num_independent:
                    change/=player_expect_vs_score[key]["num_games"]
                self.players[key]["rating"] += change

        for key in numerator_and_games:
            if not self.players[key]["rating_fixed"]:
                self.players[key]["rating"] = numerator_and_games[key]["numerator"]/numerator_and_games[key]["num_games"]

        # print("new elos",self.get_rating_table())

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

    def load_a_model_player(self,checkpoint,model_identifier,model_name=None):
        if model_name is None:
            model_name = os.path.basename(checkpoint)
        stuff = torch.load(checkpoint)
        model = get_pre_defined(model_identifier,stuff["args"]).to(device)
        model.load_state_dict(stuff["state_dict"])
        self.add_player(name=model_name,model=model,simple=False,uses_empty_model=False)

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
                            actions = [datas[i].backmap[actions[i]].item() for i in range(len(actions))]
                        elif move_num>0 or starting_moves is None:
                            if self.players[current_player]["simple"]:
                                board_actions = self.players[current_player]["model"](games)
                                actions = [game.board.board_index_to_vertex[action] for game,action in zip(games,board_actions)]
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
                                try:
                                    games[i].make_move(action,remove_dead_and_captured=True)
                                except Exception as e:
                                    print(games[i].board.to_sgf())
                                    with open("test.sgf","w") as f:
                                        f.write(games[i].board.to_sgf())
                                    print(i)
                                    print(board_actions[i])
                                    print(games[i].board.position[board_actions[i]])
                                    print(games[i].board.position)
                                    print(games[i].board.draw_me())
                                    print(games[i].who_won())
                                    games[i].draw_me()
                                    print(action)
                                    print(games[i].view.num_vertices())
                                    print(games[i].view.vertex_index.copy().fa)
                                    raise Exception(e)

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


def random_player(games):
    # print(batch.ptr[1:]-batch.ptr[:-1])
    return [game.board.sample_legal_move() for game in games]

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

def test_some_statistics():
    e = Elo_handler(5)
    e.add_player(name="random",set_rating=0,rating_fixed=True)
    e.add_player(name="maker",set_rating=1500,rating_fixed=False)
    e.add_player(name="huff",set_rating=None,rating_fixed=False)
    e.score_some_statistics([
        {"random":6,"maker":6},
        {"huff":2,"maker":10},
        {"huff":4,"random":8},
    ])
    print(e.get_rating_table())

def run_balanced_eval_roundrobin(hex_size,folder,num_from_folder=None,model_name="modern_two_headed",additonal_players=[],starting_game_frame=0,final_game_frame=np.inf,device="cpu"):
    checkpoints = [os.path.join(folder,x) for x in os.listdir(folder) if starting_game_frame<=int(os.path.basename(x).split("_")[1].split(".")[0])<=final_game_frame]
    checkpoints.sort(key=lambda x:int(os.path.basename(x).split("_")[1].split(".")[0]))
    if num_from_folder is not None and num_from_folder<len(checkpoints):
        ccs = []
        for i in np.linspace(0,len(checkpoints)-1,num_from_folder):
            ccs.append(checkpoints[int(i)])
        checkpoints = ccs

    some_stuff = torch.load(checkpoints[0])
    empty_model_func = lambda :get_pre_defined(model_name,some_stuff["args"])
    e = Elo_handler(hex_size,empty_model_func,k=1,device=device)
    for c in checkpoints:
        e.add_player(name=os.path.basename(c).split("_")[1].split(".")[0],checkpoint=c,set_rating=0,episode_number=int(os.path.basename(c).split("_")[1].split(".")[0]),uses_empty_model=True)
    for p in additonal_players:
        e.add_player(name=p["name"],model=p["model"],simple=p["simple"],set_rating=p["rating"],rating_fixed=p["rating_fixed"],uses_empty_model=False)
    e.roundrobin(num_players=None,num_games_per_match=None,score_as_n_games=1000)
    col,data = e.get_rating_table()
    df = pd.DataFrame(data=data,columns=col)
    print(df)
    df.to_csv("rating_table.csv")

def just_run_1v1(hex_size,model1_identifier,model2_identifier,checkpoint1,checkpoint2,model1_name=None,model2_name=None,device="cpu"):
    e = Elo_handler(hex_size,k=1,device=device)
    e.load_a_model_player(checkpoint1,model1_identifier,model1_name)
    e.load_a_model_player(checkpoint2,model2_identifier,model2_name)
    print(e.play_some_games(model1_name,model2_name,None,0))
    print(e.play_some_games(model2_name,model1_name,None,0))



if __name__ == "__main__":
    from Rainbow.common.utils import get_highest_model_path
    # test_some_statistics()
    hex_size=7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # old_model_stuff = torch.load(f"Rainbow/checkpoints/misty-firebrand-26/{hex_size}/checkpoint_3200000.pt")
    old_model_stuff = torch.load(f"Rainbow/checkpoints/misty-firebrand-26/{hex_size}/checkpoint_20000000.pt")
    old_model = get_pre_defined("two_headed",old_model_stuff["args"]).to(device)
    old_model.load_state_dict(old_model_stuff["state_dict"])
    old_player = {"model":old_model,"rating":0,"rating_fixed":False,"simple":False,"name":"old_model"}
    random_dude = {"name":"random","model":random_player,"simple":True,"rating":0,"rating_fixed":True}
    # folder = "Rainbow/checkpoints/fresh-wood-2188"
    folder = "Rainbow/checkpoints/sweet-plasma-2191"
    starting_frame = 9684480
    final_frame = 16634884
    

    # just_run_1v1(7,"pna_two_headed","modern_two_headed","Rainbow/checkpoints/fresh-wood-2188/checkpoint_23992320.pt","Rainbow/checkpoints/sweet-plasma-2191/checkpoint_16494720.pt",model1_name="pna",model2_name="sage")
    e = Elo_handler(11,k=1,device=device)
    e.load_a_model_player(get_highest_model_path("misty-firebrand-26/11"),"two_headed","misty-firebrand")
    e.add_player(name="random",model=random_player,set_rating=None,uses_empty_model=False,simple=True)
    e.add_player(name="mohex",model=MohexPlayer(max_time=1),set_rating=None,uses_empty_model=False,simple=True)
    res1 = e.play_some_games("misty-firebrand","mohex",None,0,progress=True)
    print(res1)
    res2 = e.play_some_games("mohex","misty-firebrand",None,0,progress=True)
    print(res1,res2)
    # run_balanced_eval_roundrobin(hex_size=hex_size,folder=folder,num_from_folder=10,model_name="modern_two_headed",additonal_players=[old_player,random_dude],starting_game_frame=starting_frame,final_game_frame=final_frame,device=device)
    # test_some_more_statistics()
    # elo_handler = Elo_handler(9)
    # checkpoint = "Rainbow/checkpoints/worldly-fire-19/checkpoint_4499712.pt"
    # model = get_pre_defined("sage+norm")
    # evaluate_checkpoint_against_random_mover(elo_handler,checkpoint,model)
    # run_league("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/ethereal-glitter-22")
    # test_elo_handler()
    # battle_it_out()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # battle_it_out(device=device)
    # old_vs_new(old_breaker_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_breaker_32800000.pt",old_maker_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_maker_32800000.pt",old_model_name="sage+norm",new_model_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/azure-snowball-157/checkpoint_59200000.pt",new_model_name="two_headed")
