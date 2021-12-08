from graph_tool.all import *
import pickle
import sys,os
import math
import time

class Board_game():
    winsquarenums:set
    position:list
    squares:int
    onturn:bool
    wp_map_rev:dict
    node_map_rev:dict

    def __init__(self):
        self.onturn = "b"
        self.node_map = dict()
        self.wp_map = dict()
        self.psets = {"bp":set(),"bd":set(),"wp":set(),"wd":set()}
    
    def inv_maps(self):
        self.wp_map_rev = {value:key for key,value in self.wp_map.items()}
        self.node_map_rev = {value:key for key,value in self.node_map.items()}

    def pos_from_graph(self):
        known_nods = [self.node_map[int(x)] for x in self.game.view.vertices() if self.game.owner_map[self.game.view.vp.o[x]] is None]
        pos = ["A"]*self.squares
        for key,value in self.wp_map.items():
            try:
                owner = self.game.owner_map[self.game.view.vp.o[self.game.view.vertex(key)]]
            except ValueError:
                continue
            for sq in value:
                if sq in known_nods:
                    pos[sq] = "f"
                else:
                    pos[sq] = owner
        return pos

    def load_set_folder(self,folder):
        self.psets = Board_game.load_psets(self.psets.keys(),folder)

    @staticmethod
    def load_psets(setnames,folder):
        psets = {setname:set() for setname in setnames}
        for key in psets:
            try:
                with open(os.path.join(folder,key+".pkl"),"rb") as f:
                    psets[key] = pickle.load(f)
            except FileNotFoundError as e:
                print(e)
        return psets

    def make_move(self, move):
        self.position[move] = self.onturn
        self.onturn = "b" if self.onturn == "w" else "b"
        self.game.graph_from_board()      
        self.create_node_hash_map()

    def set_position(self,pos,onturn):
        self.position = pos
        self.onturn = onturn
        self.game.graph_from_board()

    def draw_me(self,pos=None):
        root = int(math.sqrt(self.squares))
        out_str = "#"*(root+2)
        out_str+="\n"
        pos = self.position if pos is None else pos
        for row in range(root):
            out_str+="#"
            for col in range(root):
                out_str += " " if pos[col+row*root]=="f" else pos[col+row*root]
            out_str+="#\n"
        out_str += "#"*(root+2)
        print(out_str)
        return out_str

    def get_blocked_squares(self,ruleset):
        self.inv_maps()
        blocked_moves = set(self.node_map_rev[x] for x in self.rulesets[ruleset])
        return blocked_moves