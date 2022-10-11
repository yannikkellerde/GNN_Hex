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
import json


class Elo_handler():
    def __init__(self,hex_size,empty_model_func=None,device="cpu",k=3):
        self.players = {}
        self.size = hex_size
        self.elo_league_contestants = list()
        self.device = device
        self.K = k
        if empty_model_func is not None:
            self.create_empty_models(empty_model_func)

    def create_empty_models(self,empty_model_func):
        self.empty_model1 = empty_model_func().to(self.device)
        self.empty_model1.eval()
        self.empty_model2 = empty_model_func().to(self.device)
        self.empty_model2.eval()

    def add_player(self,name,model,set_rating=1500,simple=False,original_model=False):
        self.players[name] = {"model":model,"simple":simple,"rating":set_rating,"original_model":original_model}

    def load_into_empty_model(self,empty_model,checkpoint):
        stuff = torch.load(checkpoint,map_location=self.device)
        empty_model.load_state_dict(stuff["state_dict"])
        if "cache" in stuff and stuff["cache"] is not None:
            empty_model.import_norm_cache(*stuff["cache"])
        else:
            print("Warning, no cache")

    def run_tournament(self,players,add_to_elo_league=False,set_rating=1500,progress=False):
        for player in players:
            if player["name"] not in self.players:
                self.add_player(player["name"],player["model"] if "model" in player else self.empty_model1,set_rating=set_rating,original_model="model" in player)
        
        all_stats = []
        with alive_bar(len(players)*(len(players)-1),disable=True) as bar:
            for player1 in players:
                if not "model" in player1:
                    self.players[player1["name"]]["model"] = self.empty_model1
                if "checkpoint" in player1:
                    self.load_into_empty_model(self.players[player1["name"]]["model"],player1["checkpoint"])
                for player2 in players:
                    if player1==player2:
                        continue
                    if not "model" in player2:
                        self.players[player2["name"]]["model"] = self.empty_model2
                    if "checkpoint" in player2:
                        self.load_into_empty_model(self.players[player2["name"]]["model"],player2["checkpoint"])
                    statistics = self.play_some_games(player1["name"],player2["name"],None,0,random_first_move=True,progress=progress)
                    all_stats.append(statistics)
                    bar()

        self.score_some_statistics(all_stats)
        if add_to_elo_league:
            for player in players:
                self.elo_league_contestants.append(player.copy())
            self.elo_league_contestants.sort(key=lambda x:-self.get_rating(x["name"]))


    def add_elo_league_contestant(self,name,checkpoint,model=None,simple=False):
        if model is None:
            self.load_into_empty_model(self.empty_model1,checkpoint)
            self.add_player(name,self.empty_model1,set_rating=1500,simple=simple)
        else:
            self.load_into_empty_model(model,checkpoint)
            self.add_player(name,model,set_rating=1500)

        all_stats = []
        for contestant in self.elo_league_contestants[:10]:
            if "model" in contestant and contestant["model"] is not None:
                self.players[contestant["name"]]["model"] = contestant["model"]
            else:
                self.load_into_empty_model(self.empty_model2,contestant["checkpoint"])
                self.players[contestant["name"]]["model"] = self.empty_model2
            statistics = self.play_some_games(name,contestant["name"],None,0,random_first_move=True)
            all_stats.append(statistics)
            statistics = self.play_some_games(contestant["name"],name,None,0,random_first_move=True)
            all_stats.append(statistics)
        self.score_some_statistics(all_stats,firsto=name)
        self.elo_league_contestants.append({"name":name,"checkpoint":checkpoint,"model":model})
        self.elo_league_contestants.sort(key=lambda x:-self.get_rating(x["name"]))
        return self.get_rating(name)

    def score_some_statistics(self,statistics,firsto=None):
        player_expect_vs_score = defaultdict(lambda :dict(expectation=0,score=0,num_games=0))
        if firsto is not None:
            expectation = 0
            score = 0
            for stats in statistics:
                keys = list(stats.keys())
                if firsto in keys:
                    if keys[0] == firsto:
                        not_firsto = keys[1]
                    else:
                        not_firsto = keys[0]
                    num_games = sum([int(x) for x in stats.values()])
                    expectation += (1/(1+10**((self.get_rating(not_firsto)-self.get_rating(firsto))/400)))*num_games
                    score += stats[firsto]
            self.players[firsto]["rating"] += self.K*(score-expectation)


        for stats in statistics:
            keys = list(stats.keys())
            num_games = sum([int(x) for x in stats.values()])
            player_expect_vs_score[keys[0]]["expectation"] += (1/(1+10**((self.get_rating(keys[1])-self.get_rating(keys[0]))/400)))*num_games
            player_expect_vs_score[keys[1]]["expectation"] += (1/(1+10**((self.get_rating(keys[0])-self.get_rating(keys[1]))/400)))*num_games
            player_expect_vs_score[keys[0]]["score"] += stats[keys[0]]
            player_expect_vs_score[keys[1]]["score"] += stats[keys[1]]

        if firsto is not None and firsto in player_expect_vs_score:
            del player_expect_vs_score[firsto]
        
        for key in player_expect_vs_score:
            self.players[key]["rating"] += self.K*(player_expect_vs_score[key]["score"]-player_expect_vs_score[key]["expectation"])


    def get_rating_table(self):
        columns = ["name","rating"]
        data = []
        for contestant in self.elo_league_contestants:
            data.append([contestant["name"],self.get_rating(contestant["name"])])
        data.sort(key=lambda x:-x[1])
        return columns,data

    def get_rating(self,player_name):
        return self.players[player_name]["rating"]

    def play_some_games(self,maker,breaker,num_games,temperature,random_first_move=False,progress=False):
        print("Playing games between",maker,"and",breaker)
        wins = {maker:0,breaker:0}
        game_lengths = []
        starting_moves = None
        if num_games is None:
            some_game = Hex_game(self.size)
            starting_moves = some_game.get_unique_starting_moves()
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
                        datas = [convert_node_switching_game(game.view,global_input_properties=[game.view.gp["m"]], need_backmap=True) for game in games]
                        batch = Batch.from_data_list(datas)

                        if move_num == 0 and random_first_move and starting_moves is None:
                            actions = [random.choice(g.get_actions()) for g in games]
                        elif move_num>0 or starting_moves is None:
                            if self.players[current_player]["simple"]:
                                actions = self.players[current_player]["model"](batch)
                            else:
                                action_values = self.players[current_player]["model"].simple_forward(batch.to(self.device)).to(device)
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
                            actions = starting_moves


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

def multi_model_battle(model_names,size=5):
    paths = [get_highest_model_path(m) for m in model_names]
    players = []
    elo = Elo_handler(size,device=device,k=1)
    for name,path in zip(model_names,paths):
        stuff = torch.load(path,map_location=device)
        args = stuff["args"]
        model = get_pre_defined(args.model_name,args).to(device)
        if "cache" in stuff and stuff["cache"] is not None:
            model.import_norm_cache(*stuff["cache"])
        players.append({"name":name,"checkpoint":path,"model":model})
        print("Evaluating",name,"against random mover")
        evaluate_checkpoint_against_random_mover(elo,path,model)
    elo.run_tournament(players,add_to_elo_league=True,progress=True) 
    print(elo.get_rating_table())

def battle_it_out(device="cpu"):
    basepath = os.path.dirname(get_highest_model_path("azure-snowball-157"))
    # table_path = "/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/wandb/run-20220924_093459-23kjabmr/files/media/table/ev/rating_table_38100_87ac68f0e7970015c449.table.json"
    # with open(table_path,"r") as f:
    #     table = json.load(f)
    # cps = [int(x[0].split("_")[0]) for x in table["data"][:10]]
    players = []
    cps = [71840000,53440000,15680000,13120000,11040000,10400000]
    for cp in cps:
        path=os.path.join(basepath,f"checkpoint_{cp}.pt")
        stuff = torch.load(path,map_location=device)
        args = stuff["args"]
        model = get_pre_defined(args.model_name,args).to(device)
        if "cache" in stuff and stuff["cache"] is not None:
            model.import_norm_cache(*stuff["cache"])
        players.append({"name":cp,"checkpoint":path,"model":model})
    elo = Elo_handler(8,device=device)
    elo.run_tournament(players,add_to_elo_league=True,progress=True)
    print(elo.get_rating_table())
    print([[x["name"],elo.get_rating(x["name"])] for x in players])

def random_player(batch):
    # print(batch.ptr[1:]-batch.ptr[:-1])

    actions = [random.randint(2,after-before-1) for before,after in zip(batch.ptr,batch.ptr[1:])]
    return actions

def evaluate_elo_between(elo_handler:Elo_handler,model1,model2,checkpoint1,checkpoint2):
    stuff = torch.load(checkpoint1)
    model1.load_state_dict(stuff["state_dict"])
    model1.import_norm_cache(*stuff["cache"])
    model1.eval()
    model1.cpu()

    stuff = torch.load(checkpoint1)
    model2.load_state_dict(stuff["state_dict"])
    model2.import_norm_cache(*stuff["cache"])
    model2.eval()
    model2.cpu()

    elo_handler.add_player("model1",model1)
    elo_handler.add_player("model2",model2)
    res= elo_handler.play_some_games("model1","model2",num_games=256,temperature=0,random_first_move=True)
    print(res)

def maker_breaker_model(maker_model,breaker_model):
    class mb_model(torch.nn.Module):
        def __init__(self,mm,bm):
            super().__init__()
            self.maker_model = mm
            self.breaker_model = bm
        def forward(self,*args):
            if args[0][0,2] == 1:
                return self.maker_model(*args)
            else:
                return self.breaker_model(*args)
        def simple_forward(self,data):
            return self.forward(data.x,data.edge_index)

    return mb_model(maker_model,breaker_model)

def old_vs_new(old_breaker_path,old_maker_path,old_model_name,new_model_name,new_model_path):
    elo_handler = Elo_handler(hex_size=11)
    breaker = get_pre_defined(old_model_name).to(device)
    maker = get_pre_defined(old_model_name).to(device)
    stuff_breaker = torch.load(old_breaker_path,map_location=device)

    breaker.load_state_dict(stuff_breaker["state_dict"])
    if "cache" in stuff_breaker and stuff_breaker["cache"] is not None:
        breaker.import_norm_cache(*stuff_breaker["cache"])
    breaker.eval()
    stuff_maker = torch.load(old_maker_path,map_location=device)
    maker.load_state_dict(stuff_maker["state_dict"])
    if "cache" in stuff_maker and stuff_maker["cache"] is not None:
        maker.import_norm_cache(*stuff_maker["cache"])
    maker.eval()
    model1 = maker_breaker_model(maker,breaker)


    stuff = torch.load(new_model_path,map_location=device)
    args = stuff["args"]
    model2 = get_pre_defined(new_model_name,args).to(device)
    model2.load_state_dict(stuff["state_dict"])
    if "cache" in stuff and stuff["cache"] is not None:
        model2.import_norm_cache(*stuff["cache"])
    model2.eval()

    elo_handler.add_player("random",random_player,simple=True)
    elo_handler.add_player("old",model1)
    elo_handler.add_player("new",model2)
    stats = []
    res= elo_handler.play_some_games("old","random",num_games=64,temperature=0,random_first_move=True,progress=True)
    print(res)
    stats.append(res)
    res= elo_handler.play_some_games("new","random",num_games=64,temperature=0,random_first_move=True,progress=True)
    print(res)
    stats.append(res)
    res= elo_handler.play_some_games("random","old",num_games=64,temperature=0,random_first_move=True,progress=True)
    print(res)
    stats.append(res)
    res= elo_handler.play_some_games("random","new",num_games=64,temperature=0,random_first_move=True,progress=True)
    print(res)
    stats.append(res)
    res= elo_handler.play_some_games("old","new",num_games=256,temperature=0,random_first_move=True,progress=True)
    print(res)
    stats.append(res)
    res= elo_handler.play_some_games("new","old",num_games=256,temperature=0,random_first_move=True,progress=True)
    print(res)
    stats.append(res)
    print(stats)
    elo_handler.score_some_statistics(stats)

    print(f'old: {elo_handler.get_rating("old")}\nnew: {elo_handler.get_rating("new")}\nrandom: {elo_handler.get_rating("random")}')

def run_league(checkpoint_folder):
    elo_handler = Elo_handler(5,empty_model_func=lambda :get_pre_defined("sage+norm"))
    checkpoints = defaultdict(dict)
    for checkpoint_file in os.listdir(checkpoint_folder):
        splits = checkpoint_file.split("_")
        frame = int(splits[-1].split(".")[0])
        checkpoints[frame][splits[1]] = os.path.join(checkpoint_folder,checkpoint_file)
    
    for key in sorted(checkpoints):
        elo_handler.size = min((key//600000+5),11)
        res = elo_handler.add_elo_league_contestant(str(key),checkpoints[key]["maker"],checkpoints[key]["breaker"])
        print(res)

    print(elo_handler.get_rating_table())


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


def test_elo_handler():
    e = Elo_handler(5)
    e.add_player("maker",1500)
    e.add_player("breaker",1500)
    e.add_player("random",1500)
    s1 = {"maker":120,"random":8}
    s2 = {"breaker":110,"random":18}
    s3 = {"maker":80,"breaker":48}
    e.score_some_statistics([s1,s2,s3])
    print(e.get_rating("maker"),e.get_rating("breaker"),e.get_rating("random"))


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
    multi_model_battle(model_names=["misty-firebrand-26/11","misty-firebrand-26/8","misty-firebrand-26/10","azure-snowball-157","wobbly-disco-167"],size=8)
    # battle_it_out(device=device)
    # old_vs_new(old_breaker_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_breaker_32800000.pt",old_maker_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/breezy-morning-37/checkpoint_maker_32800000.pt",old_model_name="sage+norm",new_model_path="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/azure-snowball-157/checkpoint_59200000.pt",new_model_name="two_headed")
