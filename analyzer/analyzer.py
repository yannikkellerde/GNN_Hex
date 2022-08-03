import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.dirname(__file__))
from graph_tools_games import instanz_by_name
from winpattern_board import Winpattern_board
from solve_graph_tools import background_thread
from util import provide_room_num
import math
import pickle
import json
import time
from threading import Lock
from collections import defaultdict
from flask import render_template
from flask_socketio import join_room, leave_room

from functools import reduce
import os

base_path = os.path.abspath(os.path.dirname(__file__))

FILE = "python_server/explore_wins.html"
PORT = 8081

class Solver_analyze():
    def __init__(self):
        self.session_to_game_name = {}
        self.session_to_pset_name = {}
        self.proofsets_to_sessions = defaultdict(set)
        self.pset_name_to_pset = {}
        self.session_last_access = {}
        self.sid_to_thread = {}
        self.lock = Lock()
        self.session_timeout = 3600
        self.psetnames = ("bd","bp","wd","wp")
        self.ownfolders = set(("qango6x6_static","qango7x7_static","qango7x7_plus_static"))

    def timeout_sessions(self):
        now = time.time()
        delids = []
        for uid,value in self.session_last_access.items():
            if time.time() - value > 3600:
                if uid in session_to_pset_name:
                    pset_name = self.session_to_pset_name[uid]
                    del self.session_to_pset_name[uid]
                    self.proofsets_to_sessions[pset_name].remove(uid)
                    if len(self.proofsets_to_sessions[pset_name])==0:
                        del self.pset_name_to_pset[pset_name]
                delids.append(uid)
        for delid in delids:
            del self.session_last_access[delid]

    def save_callback(self,psetname,color,proofset,disproofset):
        if psetname in self.pset_name_to_pset:
            self.pset_name_to_pset[psetname][color+"p"] = proofset
            self.pset_name_to_pset[psetname][color+"d"] = disproofset
        with open(os.path.join(base_path,"../proofsets",psetname,color+"p.pkl"),"wb") as f:
            pickle.dump(proofset,f)
        with open(os.path.join(base_path,"../proofsets",psetname,color+"d.pkl"),"wb") as f:
            pickle.dump(disproofset,f)

    def start_search(self,data,sid,uid,socketio):
        color = data["color"]
        blocked = data["blocked_sq"]
        if uid in self.session_to_game_name:
            game_name = self.session_to_game_name[uid]
        else:
            game = self.get_game(data["game_name"],uid)
        room_num = provide_room_num()
        if uid not in self.session_to_pset_name:
            self.get_proofsets(data["pset_name"],uid)
        psetname = self.session_to_pset_name[uid]
        if psetname not in self.pset_name_to_pset:
            self.get_proofsets(data["pset_name"],uid)
            psetname = self.session_to_pset_name[uid]
        psets = self.pset_name_to_pset[psetname]
        join_room(room_num,sid=sid)
        save_callback = lambda pset,dset:self.save_callback(psetname,color,pset,dset)
        back_thread = lambda :background_thread(color,sid,room_num,game_name,data["position"],"b" if data["onturn"]==2 else "w",blocked,psets,save_callback,socketio)
        with self.lock:
            self.sid_to_thread[sid] = socketio.start_background_task(target=back_thread)
        #back_thread()

    def get_proofsets(self,proofsetname,uid):
        if not (uid in self.session_to_pset_name and proofsetname == self.session_to_pset_name[uid]):
            if proofsetname not in self.pset_name_to_pset:
                self.pset_name_to_pset[proofsetname] = Winpattern_board.load_psets(self.psetnames,os.path.join(base_path,"../proofsets",proofsetname))
            self.session_to_pset_name[uid] = proofsetname
            self.proofsets_to_sessions[proofsetname].add(uid)
        return self.pset_name_to_pset[self.session_to_pset_name[uid]]

    def get_game(self,game_name,uid):
        if not (uid in self.session_to_game and game_name in self.game_name_to_game and
                self.game_name_to_game[game_name] == self.session_to_game[uid]):
            if game_name not in self.game_name_to_game:
                game = instanz_by_name(game_name)
                self.game_name_to_game[game_name] = game
            self.session_to_game[uid] = self.game_name_to_game[game_name]
            self.game_name_to_sessions[game_name].add(uid)
        return self.session_to_game[uid]

    def do_GET(self,game_name,uid):
        self.session_last_access[uid] = time.time()
        self.timeout_sessions()
        self.session_to_game_name[uid] = game_name
        folder_name = game_name if game_name in self.ownfolders else "json"
        with open(os.path.join(base_path,folder_name,"board.html"),"r") as f:
            my_board = f.read()
        return render_template(os.path.join("explore_wins.html"),board = my_board, stylesheet=os.path.join("/static_analyzer",folder_name,"board.css"),
                               game_js_path=os.path.join("/static_analyzer",folder_name,"game.js"))

    def create_proofset(self,game,new):
        path = os.path.join(base_path,"..","proofsets",new)
        if not os.path.exists(path):
            os.mkdir(path)
            for setname in game.board.psets:
                with open(os.path.join(path,setname+".pkl"), 'wb') as f:
                    pickle.dump(set(),f)

    def do_POST(self,data,uid):
        self.session_last_access[uid] = time.time()
        out = json.dumps({"error":"NOT FOUND"})
        game = instanz_by_name(data["game_name"])
        print(data)
        if "request" in data:
            if data["request"] == "config":
                out = json.dumps(game.config)
            if data["request"] == "rulesets":
                out = json.dumps(game.board.rulesets)
            if data["request"] == "aval_proofsets":
                proofsets = os.listdir(os.path.join(base_path,"..","proofsets"))
                if game.name in proofsets:
                    pname = game.name
                    self.get_proofsets(game.name,uid)
                else:
                    if uid in self.session_to_pset_name:
                        pname = self.session_to_pset_name[uid]
                    else:
                        if len(proofsets)>0:
                            pname = proofsets[0]
                        else:
                            pname = "new_proofset"
                out = json.dumps({"proofsets":proofsets,"default":pname if pname in proofsets else proofsets[0]})
        elif "new_proofset" in data:
            self.create_proofset(game,data["new_proofset"])
            proofsets = os.listdir(os.path.join(base_path,"..","proofsets"))
            self.get_proofsets(data["new_proofset"],uid)
            out = json.dumps({"proofsets":proofsets,"default":data["new_proofset"],"changed_to":data["new_proofset"]})
        elif "set_proofset" in data:
            self.get_proofsets(data["set_proofset"],uid)
            out = json.dumps({"changed_to":data["set_proofset"]})
        elif "position" in data:
            if uid in self.session_to_pset_name and self.session_to_pset_name[uid]==data["pset_name"]:
                game.board.psets = self.pset_name_to_pset[self.session_to_pset_name[uid]]
            else:
                game.board.psets = self.get_proofsets(data["pset_name"],uid)
            game.board.psets = self.pset_name_to_pset[self.session_to_pset_name[uid]]
            real_pos = [("f" if x==0 else ("b" if x==2 else "w")) for x in data["position"]]
            game.board.set_position(real_pos,"b" if data["onturn"]==2 else "w")
            game.board.draw_me()
            depth = len(list(filter(lambda x:x!="f",real_pos)))
            moves = game.get_actions(filter_superseeded=False,none_for_win=False)
            board_moves = [game.board.node_map[x] for x in moves]
            game.draw_me()
            evals = game.check_move_val(moves)
            moves_with_eval = list(zip(board_moves, evals))
            out = json.dumps({"moves":moves_with_eval})
        elif "new_rule" in data:
            game.board.rulesets[data["new_rule"]] = data["blocked"]
            with open(os.path.join(base_path,"../rulesets",game.name+".json"),"w") as f:
                json.dump(game.board.rulesets,f)
            out = json.dumps(game.board.rulesets)
        elif "del_rule" in data:
            if (len(game.board.rulesets)>1):
                del game.board.rulesets[data["del_rule"]]
                with open(os.path.join(base_path,"../rulesets",game.name+".json"),"w") as f:
                    json.dump(game.board.rulesets,f)
            out = json.dumps(game.board.rulesets)
        return out

analyze_handler = Solver_analyze()
