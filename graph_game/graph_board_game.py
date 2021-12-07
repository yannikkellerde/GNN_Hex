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

    def check_move_val(self,moves,priorize_sets=True):
        """{"-4":"White wins (Forced Move)","-3":"White wins (Threat search)","-2":"White wins (Proofset)",
         "-1":"White wins or draw","u":"Unknown",0:"Draw",1:"Black wins or draw",2:"Black wins (Proofset)",
         3:"Black wins (Threat search)",4:"Black wins (Forced Move)"}"""
        self.inv_maps()
        winmoves = self.game.win_threat_search(one_is_enough=False,until_time=time.time()+5)
        self.game.view.gp["b"] = not self.game.view.gp["b"]
        defense_vertices,has_threat,_ = self.game.threat_search()
        self.game.view.gp["b"] = not self.game.view.gp["b"]
        results = []
        storage = self.game.extract_storage()
        for move in moves:
            val = "u"
            self.game.load_storage(storage)
            if has_threat and move not in defense_vertices and move not in winmoves:
                if self.game.onturn=="b":
                    val = -3
                else:
                    val = 3
            else:
                self.game.make_move(move)
                self.game.hashme()
                if self.game.hash in self.psets["wp"]:
                    val = -2
                elif self.game.hash in self.psets["bp"]:
                    val = 2
                elif self.game.hash in self.psets["wd"]:
                    val = 1
                if self.game.hash in self.psets["bd"]:
                    if val == 1:
                        val = 0
                    elif val =="u":
                        val = -1
                if val=="u" or not priorize_sets:
                    if self.game.view.num_vertices() == 0:
                        val = 0
                    else:
                        if move in winmoves:
                            if self.game.onturn=="b":
                                val = -4
                            else:
                                val = 4
                        else:
                            movs = len(self.game.win_threat_search(one_is_enough=True,until_time=time.time()+0.5))>0
                            if movs:
                                if self.game.onturn=="b":
                                    val = 4
                                else:
                                    val = -4
            results.append(val)
        return results

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