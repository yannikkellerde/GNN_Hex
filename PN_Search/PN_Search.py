import math
from util import resources_avaliable
import pickle
import os,sys
from graph_tools_games import instanz_by_name
from graph_tools_game import Graph_game
from graph_tool.all import *
import numpy as np
import time
from typing import Callable
import threading
import json
from flask_socketio import emit, send

# Node storage for memory efficency in lists
PN = 0 # int
DN = 1 # int
HASH = 2 # int
PARENTS = 3 # List of Nodes(which are lists)
CHILDREN = 4 # List of tuples containing move made and Node (which is a lists)
PROOFNODE = 5 # Bool, are we in a proof node or disproof node
STORAGE = 6 # A tuple containing 1. an owner map, 2. a filter map, 3. onturn bool


class PN_search():
    def __init__(self, game:Graph_game, callback:Callable, save_callback:Callable, drawproves=False):
        self.game = game
        self.ttable = {}
        self.provenset = set()
        self.disprovenset = set()
        self.node_count = 0
        self.alive_graphs = 0
        self.proofadds = [0,0]
        self.drawproves = drawproves
        self.callback = callback
        self.callback_it = 100
        self.save_callback = save_callback
        self.save_it = 10000
        self.runtime_start = time.time()
    
    def set_pn_dn(self, n):
        if n[PN] == 0 or n[DN] == 0:
            return
        if n[PROOFNODE]:
            n[PN] = math.inf; n[DN] = 0
            for c in n[CHILDREN]:
                n[DN] += c[DN]
                n[PN] = min(n[PN], c[PN])
        else:
            n[DN] = math.inf; n[PN] = 0
            for c in n[CHILDREN]:
                n[PN] += c[PN]
                n[DN] = min(n[DN], c[DN])

    def delete_node(self, n, ps, ch):
        if n==self.root:
            return
        for p in ps:
            for i in range(len(p[CHILDREN])):
                if p[CHILDREN][i]==n:
                    del p[CHILDREN][i]
                    break
        for c in ch:
            if len(c)>PARENTS:
                c[PARENTS].remove(n)
                if len(c[PARENTS])==0:
                    self.delete_node(c, [], c[CHILDREN].copy())
        del self.ttable[n[HASH]]
        if n[PN] == 0:
            self.provenset.add(n[HASH])
            self.proofadds[0] += 1
        elif n[DN] == 0:
            self.disprovenset.add(n[HASH])
            self.proofadds[1] += 1
        if len(n)>STORAGE:
            self.alive_graphs-=1
            del n[STORAGE]
        del n[PROOFNODE]
        del n[CHILDREN]
        del n[PARENTS]
        # Do not use n aferwards
    
    def update_anchestors(self, n):
        old_pn = n[PN]
        old_dn = n[DN]
        self.set_pn_dn(n)
        if n[PN] == old_pn and n[DN] == old_dn and n[PN]!=0 and n[DN]!=0:
            return
        for p in n[PARENTS].copy():
            if len(p)>3:
                self.update_anchestors(p)
        if (n[PN] == 0 or n[DN] == 0) and len(n)>PARENTS:
            self.delete_node(n, n[PARENTS], n[CHILDREN])

    def select_most_proving(self, n):
        depth = 0
        while n[CHILDREN]:
            val = math.inf
            best = None
            if n[PROOFNODE]:
                for child in n[CHILDREN]:
                    if val > child[PN]:
                        best = child
                        val = child[PN]
            else:
                for child in n[CHILDREN]:
                    if val > child[DN]:
                        best = child
                        val = child[DN]
            n = best
            depth += 1
        return n,depth

    def evaluate(self,n,hashval):
        if hashval in self.provenset:
            return True
        elif hashval in self.disprovenset:
            return False
        elif self.game.view.num_vertices()==0:
            return self.drawproves
        return None

    def expand(self, n, threat_search=True, blocked_moves=None):
        self.game.load_storage(n[STORAGE])
        moves = self.game.get_actions(filter_superseeded=blocked_moves is None)
        if moves is None:
            if n[PROOFNODE]:
                n[PN] = 0
                n[DN] = math.inf
            else:
                n[PN] = math.inf
                n[DN] = 0
            del n[STORAGE]
            return
        if len(moves)==0:
            if self.drawproves:
                n[PN] = 0
                n[DN] = math.inf
            else:
                n[PN] = math.inf
                n[DN] = 0
            del n[STORAGE]
            return
        if blocked_moves is not None:
            moves = [move for move in moves if move not in blocked_moves]
        if threat_search:
            self.game.view.gp["b"] = not self.game.view.gp["b"]
            defense_vertices,has_threat,_ = self.game.threat_search()
            self.game.view.gp["b"] = not self.game.view.gp["b"]
            if has_threat:
                moves = [move for move in moves if move in defense_vertices]
        knownhashvals = set()
        for move in moves:
            if move != moves[0]:
                self.game.load_storage(n[STORAGE])
            self.game.make_move(move)
            self.game.hashme()
            hashval = self.game.hash
            if hashval in knownhashvals:
                continue
            if hashval in self.ttable:
                found = self.ttable[hashval]
                found[PARENTS].append(n)
                n[CHILDREN].append(found)
                continue
            knownhashvals.add(hashval)
            res = self.evaluate(n,hashval)
            if res is None:
                child = [1,1,hashval,[n],[],not n[PROOFNODE],self.game.extract_storage()]
                self.alive_graphs += 1
                self.ttable[hashval]=child
            else:
                if res:
                    if not n[PROOFNODE]:
                        continue
                    child = [0,math.inf]
                else:
                    if n[PROOFNODE]:
                        continue
                    child = [math.inf,0]
            n[CHILDREN].append(child)
            self.node_count += 1
            if res==n[PROOFNODE]:
                break
        del n[STORAGE]
        self.alive_graphs -= 1

    def pn_search(self,onturn_proves=True,verbose=True,save=True,ruleset=2):
        blocked = self.game.board.get_blocked_squares(ruleset)
        prove_color = self.game.onturn if onturn_proves else ("w" if self.game.onturn=="b" else "b")
        print("***********************\n",self.game.board.pos_from_graph(),self.game.onturn,prove_color,"\n**********************")
        self.game.hashme()
        hashval = self.game.hash
        self.root = [1,1,hashval,[],[],onturn_proves,self.game.extract_storage()]
        self.alive_graphs+=1
        self.node_count += 1
        self.ttable[hashval] = self.root
        c = 1
        times = {"select_most_proving":[],"expand":[],"update_anchestors":[],"whole_it":[]}
        starts = {}
        while self.root[PN]!=0 and self.root[DN]!=0:
            if verbose:
                if c % self.callback_it == 1:
                    data = {}
                    data["iteration"] = c
                    data["node_count"] = self.node_count
                    data["alive_graphs"] = self.alive_graphs
                    data["runtime"] = time.time()-self.runtime_start
                    cur = self.root
                    depth = 0
                    while len(cur[CHILDREN]) == 1:
                        cur = cur[CHILDREN][0]
                        depth += 1
                    data["depth"] = depth
                    data["PNs"] = [x[PN] for x in cur[CHILDREN]]
                    data["DNs"] = [x[DN] for x in cur[CHILDREN]]
                    data["proofadds"] = self.proofadds
                    data["recently_saved"] = c%self.save_it==1
                    aval = resources_avaliable()
                    if not self.callback(data) or not aval:
                        self.callback({"failed":{"proofadds":self.proofadds,"PN":self.root[PN], "DN":self.root[DN], "runtime":time.time()-self.runtime_start, "no_resource":not aval}})
                        return False
            if save:
                if c%self.save_it==0:
                    self.save_callback(self.provenset,self.disprovenset)
            c+=1
            most_proving,depth = self.select_most_proving(self.root)
            self.expand(most_proving,threat_search=True,blocked_moves=blocked if depth==0 else None)
            self.update_anchestors(most_proving)
        if save:
            self.callback({"solved":{"proofadds":self.proofadds,"PN":self.root[PN], "DN":self.root[DN], "runtime":time.time()-self.runtime_start}})
            self.save_callback(self.provenset,self.disprovenset)
        return True

def my_callback(room,socketio,data):
    t = threading.currentThread()
    if getattr(t, "do_run", True):
        socketio.emit("solve_state",json.dumps(data),room=room)
        return True
    else:
        return False

def background_thread(color,sid,room,game_name,position,onturn,blocked,psets,store_proofsets_callback,socketio):
    position = [("f" if x==0 else ("b" if x==2 else "w")) for x in position]
    game = instanz_by_name(game_name)
    game.board.set_position(position,onturn)
    game.board.psets = psets
    game.board.rulesets["temp_rules"] = blocked
    pn_s = PN_search(game,lambda data:my_callback(room,socketio,data),store_proofsets_callback)
    pn_s.provenset = game.board.psets[color+"p"]
    pn_s.disprovenset = game.board.psets[color+"d"]
    res = pn_s.pn_search(onturn_proves=color==game.onturn,ruleset="temp_rules")
    """for i in range(10):
        time.sleep(1)
        my_callback(room,socketio,{"iteration":i,"proofadds":[i,i],"node_count":i,"alive_graphs":i,"runtime":i,"depths":i,"PNs":[i],"DNs":[i],"recently_saved":False})"""
    print("PN search with status",res)