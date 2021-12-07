from graph_tools_game import Graph_game
from graph_board_game import Board_game
from PN_Search.util import findfivers, findsquares, remove_useless_wsn
from graph_tool.all import *
import json
from collections import defaultdict
import os,sys
base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")

class Json_game(Graph_game):
    def __init__(self,json_path):
        super().__init__()
        with open(json_path,"r") as f:
            self.config = json.load(f)
        self.board = Json_board(self.config)
        self.board.game = self
        self.graph_from_board()
        self.name = self.config["name"]

class Json_board(Board_game):
    def __init__(self,config):
        super().__init__()
        self.squares = config["squares"]
        self.position = ["f" for _ in range(self.squares)]
        self.winsquarenums = set(frozenset(x) for x in config["winsquarenums"])
        remove_useless_wsn(self.winsquarenums)
        with open(os.path.join(base_path,f"rulesets/{config['name']}.json"),"r") as f:
            self.rulesets = json.load(f)


class Qango6x6(Graph_game):
    def __init__(self):
        super().__init__()
        self.board = Qango6x6_board()
        self.board.game = self
        self.graph_from_board()
        self.name = "qango6x6"

class Qango6x6_board(Board_game):
    def __init__(self):
        super().__init__()
        self.squares = 36
        self.position = ["f" for _ in range(self.squares)]
        self.winsquarenums = {
            frozenset({0,1,6}),frozenset({4,5,11}),frozenset({24,30,31}),frozenset({29,34,35}),
            frozenset({2,7,12}),frozenset({3,10,17}),frozenset({18,25,32}),frozenset({23,28,33}),
            frozenset({8,13,14}),frozenset({9,15,16}),frozenset({19,20,26}),frozenset({21,22,27})
        }
        self.winsquarenums.update(findsquares(self.squares))
        self.winsquarenums.update(findfivers(self.squares))
        remove_useless_wsn(self.winsquarenums)
        with open(os.path.join(base_path,"rulesets/qango6x6.json"),"r") as f:
            self.rulesets = json.load(f)

class Tic_tac_toe(Graph_game):
    def __init__(self):
        super().__init__()
        self.board = Tic_tac_toe_board()
        self.board.game = self
        self.graph_from_board()
        self.name = "tic_tac_toe"

class Tic_tac_toe_board(Board_game):
    def __init__(self):
        super().__init__()
        self.squares = 9
        self.position = ["f" for _ in range(self.squares)]
        self.winsquarenums = {frozenset({0,1,2}),frozenset({3,4,5}),frozenset({6,7,8}),
                              frozenset({0,3,6}),frozenset({1,4,7}),frozenset({2,5,8}),
                              frozenset({0,4,8}),frozenset({2,4,6})}
        with open(os.path.join(base_path,"rulesets/tic_tac_toe.json"),"r") as f:
            self.rulesets = json.load(f)

class Qango7x7(Graph_game):
    def __init__(self):
        super().__init__()
        self.board = Qango7x7_board()
        self.board.game = self
        self.graph_from_board()
        self.name = "qango7x7"

class Qango7x7_board(Board_game):
    def __init__(self):
        super().__init__()
        self.squares = 49
        self.position = ["f" for _ in range(self.squares)]
        self.winsquarenums = {
            frozenset({0,1,7}),frozenset({5,6,13}),frozenset({35,42,43}),
            frozenset({41,47,48}),frozenset({2,8,14}),frozenset({4,12,20}),
            frozenset({28,36,44}),frozenset({34,40,46}),frozenset({3,9,10}),
            frozenset({26,27,33}),frozenset({29,30,37}),frozenset({11,18,19}),
            frozenset({15,21,22}),frozenset({38,39,45}),frozenset({16,17,23}),
            frozenset({25,31,32})
        }
        self.winsquarenums.update(findsquares(self.squares))
        self.winsquarenums.update(findfivers(self.squares))
        remove_useless_wsn(self.winsquarenums)
        with open(os.path.join(base_path,"rulesets/qango7x7.json"),"r") as f:
            self.rulesets = json.load(f)

class Qango7x7_plus(Graph_game):
    def __init__(self):
        super().__init__()
        self.board = Qango7x7_plus_board()
        self.board.game = self
        self.graph_from_board()
        self.name = "qango7x7_plus"

class Qango7x7_plus_board(Board_game):
    def __init__(self):
        super().__init__()
        self.squares = 37
        self.position = ["f" for _ in range(self.squares)]
        self.winsquarenums = {
            frozenset({2,8,14}),frozenset({4,12,20}),
            frozenset({28,36,44}),frozenset({34,40,46}),frozenset({3,9,10}),
            frozenset({26,27,33}),frozenset({29,30,37}),frozenset({11,18,19}),
            frozenset({15,21,22}),frozenset({38,39,45}),frozenset({16,17,23}),
            frozenset({25,31,32})
        }
        self.winsquarenums.update(findsquares(49))
        self.winsquarenums.update(findfivers(49))
        remove_useless_wsn(self.winsquarenums)
        self.change_wsn()
        with open(os.path.join(base_path,"rulesets/qango7x7_plus.json"),"r") as f:
            self.rulesets = json.load(f)

    def change_wsn(self):
        removals = [0,1,5,6,7,13,35,42,43,41,47,48]
        new_wsn = set()
        for wsn in self.winsquarenums:
            new = set()
            for ws in wsn:
                if ws in removals:
                    break
                for i,r in enumerate(removals):
                    if r>ws:
                        break
                unders = i
                new.add(ws-unders)
            else:
                new_wsn.add(frozenset(new))
        self.winsquarenums = new_wsn

    def draw_me(self,pos=None):
        row_starts = [2,1,0,0,0,1,2]
        row_ends = [5,6,7,7,7,6,5]
        out = "#"*(7+2)+"\n"
        count = 0
        pos = self.position if pos is None else pos
        for rs,re in zip(row_starts,row_ends):
            out+=" "*rs+"#"
            for _ in range(rs,re):
                out+=pos[count]
                count+=1
            out+="#"+" "*re+"\n"
        out += "#"*(7+2)
        print(out)
        return out

def instanz_by_name(game_name) -> Graph_game:
    if "qango6x6_static"==game_name:
        game = Qango6x6()
    elif "qango7x7_static"==game_name:
        game = Qango7x7()
    elif game_name == "qango7x7_plus_static":
        game = Qango7x7_plus()
    else:
        game = Json_game(os.path.join(base_path,"json_games",game_name+".json"))
    return game

if __name__ == "__main__":
    q = Qango7x7_board()
    print(q.winsquarenums)