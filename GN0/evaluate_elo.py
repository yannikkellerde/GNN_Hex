from elosports.elo import Elo
import os
import numpy as np
import random
from graph_game.graph_tools_games import Hex_game
from GN0.convert_graph import convert_node_switching_game
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch.distributions.categorical import Categorical
from alive_progress import alive_bar
from GN0.models import get_pre_defined
from collections import defaultdict,deque
from GN0.util import fix_size_defaultdict

class Elo_handler():
    def __init__(self,hex_size,empty_model_func=None,device="cpu"):
        self.players = {}
        self.league = Elo(k=20,homefield=0)
        self.size = hex_size
        self.elo_league_contestants = fix_size_defaultdict(dict,max=10)
        self.all_elo_league_contestants = list()
        self.device = device
        if empty_model_func is not None:
            self.empty_model1 = empty_model_func().to(device)
            self.empty_model2 = empty_model_func().to(device)
            self.empty_model1.eval()
            self.empty_model2.eval()

    def add_player(self,name,model,fix_rating=None,simple=False):
        self.players[name] = {"model":model,"simple":simple}
        if fix_rating:
            self.league.addPlayer(name,rating=fix_rating)
        else:
            self.league.addPlayer(name)

    def load_into_empty_model(self,empty_model,checkpoint):
        stuff = torch.load(checkpoint)
        empty_model.load_state_dict(stuff["state_dict"])
        if "cache" in stuff and stuff["cache"] is not None:
            empty_model.import_norm_cache(*[x.to(self.device) for x in stuff["cache"]])
        else:
            print("Warning, no cache")


    def add_elo_league_contestant(self,name,maker_checkpoint,breaker_checkpoint,simple=False):
        self.add_player(name+"_maker",self.empty_model1,fix_rating=1500,simple=simple)
        self.add_player(name+"_breaker",self.empty_model1,fix_rating=1500,simple=simple)
        for contestant in self.elo_league_contestants:
            self.load_into_empty_model(self.empty_model1,maker_checkpoint)
            self.load_into_empty_model(self.empty_model2,self.elo_league_contestants[contestant]["checkpoint_breaker"])
            self.players[contestant+"_breaker"]["model"] = self.empty_model2
            print(self.play_some_games(name+"_maker",contestant+"_breaker",64,0,random_first_move=True))
            self.load_into_empty_model(self.empty_model1,breaker_checkpoint)
            self.load_into_empty_model(self.empty_model2,self.elo_league_contestants[contestant]["checkpoint_maker"])
            self.players[contestant+"_maker"]["model"] = self.empty_model2
            print(self.play_some_games(contestant+"_maker",name+"_breaker",64,0,random_first_move=True))
        self.elo_league_contestants[name]["checkpoint_maker"] = maker_checkpoint
        self.elo_league_contestants[name]["checkpoint_breaker"] = breaker_checkpoint
        self.all_elo_league_contestants.append(name)
        return self.get_rating(name+"_maker"), self.get_rating(name+"_breaker")

    def get_rating_table(self):
        columns = ["name","rating"]
        data = []
        for contestant in self.all_elo_league_contestants:
            data.append([contestant+"_maker",self.get_rating(contestant+"_maker")])
            data.append([contestant+"_breaker",self.get_rating(contestant+"_breaker")])
        data.sort(key=lambda x:-x[1])
        return columns,data

    def get_best_contestants(self):
        best_maker = None
        best_maker_elo = 0
        best_breaker = None
        best_breaker_elo = 0
        for contestant in self.elo_league_contestants:
            maker_rating = self.get_rating(contestant+"_maker")
            breaker_rating = self.get_rating(contestant+"_breaker")
            if maker_rating > best_maker_elo:
                best_maker_elo = maker_rating
                best_maker = contestant
            
            if breaker_rating > best_breaker_elo:
                best_breaker_elo = breaker_rating
                best_breaker = contestant
        return (best_maker,best_breaker),(best_maker_elo,best_breaker_elo)


    def get_rating(self,player_name):
        return self.league.ratingDict[player_name]

    def play_some_games(self,maker,breaker,num_games,temperature,random_first_move=False):
        print("Playing games between",maker,"and",breaker)
        wins = {maker:0,breaker:0}
        game_lengths = []
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

                    if move_num == 0 and random_first_move:
                        actions = [random.randint(2,after-before-1) for before,after in zip(batch.ptr,batch.ptr[1:])]
                    else:
                        if self.players[current_player]["simple"]:
                            actions = self.players[current_player]["model"](batch)
                        else:
                            action_values = self.players[current_player]["model"].simple_forward(batch.to(self.device)).cpu()
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
                    to_del = []
                    actions = [datas[i].backmap[actions[i]].item() for i in range(len(actions))]
                    # if actions[0]!=0:
                    #     print(games[0].board.vertex_to_board_index[games[0].view.vertex(actions[0])])
                    #     print(f"Red:{p1}, Blue:{p2}, onturn: {games[0].onturn}, action: {games[0].board.vertex_index_to_string_move(actions[0])}")
                    #     print(games[0].board.draw_me())
                    for i,action in enumerate(actions):
                        if action!=0:
                            games[i].make_move(action,remove_dead_and_captured=True)
                            winner = games[i].who_won()
                            
                            if winner is not None:
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

        while wins[maker]>0 or wins[breaker]>0:
            if wins[maker]>0:
                self.league.gameOver(winner=maker,loser=breaker, winnerHome=True)
                wins[maker]-=1
            if wins[breaker]>0:
                self.league.gameOver(winner=breaker,loser=maker, winnerHome=True)
                wins[breaker]-=1
        return statistics

def random_player(batch):
    # print(batch.ptr[1:]-batch.ptr[:-1])

    actions = [random.randint(2,after-before-1) for before,after in zip(batch.ptr,batch.ptr[1:])]
    return actions

def evaluate_elo_between(elo_handler:Elo_handler,model1,model2,checkpoint1,checkpoint2):
    stuff = torch.load(checkpoint1)
    model1.load_state_dict(stuff["state_dict"])
    model1.import_norm_cache(*[x.cpu() for x in stuff["cache"]])
    model1.eval()
    model1.cpu()

    stuff = torch.load(checkpoint2)
    model2.load_state_dict(stuff["state_dict"])
    model2.import_norm_cache(*[x.cpu() for x in stuff["cache"]])
    model2.eval()
    model2.cpu()

    elo_handler.add_player("model1",model1)
    elo_handler.add_player("model2",model2)
    res = elo_handler.play_some_games("model1","model2",num_games=256,temperature=0,random_first_move=True)
    print(res)
    print(elo_handler.get_rating("model1"))
    print(elo_handler.get_rating("model2"))

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
    model.import_norm_cache(*[x.cpu() for x in stuff["cache"]])
    model.eval()
    model.cpu()
    elo_handler.add_player("model",model)
    elo_handler.add_player("model2",model)
    elo_handler.add_player("random",random_player,fix_rating=1500,simple=True)
    elo_handler.add_player("random2",random_player,fix_rating=1500,simple=True)
    # res = elo_handler.play_some_games("model","random",num_games=64,temperature=0)
    res = elo_handler.play_some_games("model","model2",num_games=64,temperature=0.0001,random_first_move=False)
    # res = elo_handler.play_some_games("random2","random",num_games=64,temperature=0)
    print(res)
    print(elo_handler.get_rating("model"))
    print(elo_handler.get_rating("random"))

if __name__ == "__main__":
    # elo_handler = Elo_handler(9)
    # checkpoint = "Rainbow/checkpoints/worldly-fire-19/checkpoint_4499712.pt"
    # model = get_pre_defined("sage+norm")
    # evaluate_checkpoint_against_random_mover(elo_handler,checkpoint,model)
    run_league("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/Rainbow/checkpoints/ethereal-glitter-22")


