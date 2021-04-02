import math
from copy import copy,deepcopy
from functools import reduce
from collections import defaultdict
import time
import numpy as np
import pickle
import time

from graph_tool.all import *
from graph_tools_hashing import wl_hash

class Graph_game():
    graph: Graph
    view: GraphView
    def __init__(self):
        self.owner_map = {0:None,1:"f",2:"b",3:"w"}
        self.owner_rev = {val:key for key,val in self.owner_map.items()}
        self.known_gain_sets = []

    @property
    def hash(self):
        return self.view.gp["h"]

    @property
    def onturn(self):
        return "b" if self.view.gp["b"] else "w"

    def hashme(self):
        wl_hash(self.view,self.view.vp.o,iterations=3)

    def load_storage(self,storage):
        self.graph.vp.f = storage[1].copy()
        self.view = GraphView(self.graph,vfilt=self.graph.vp.f)
        self.view.vp.o = storage[0].copy()
        self.view.gp["b"] = storage[2]

    def extract_storage(self):
        return (self.view.vp.o.copy(),self.view.vp.f.copy(),self.view.gp["b"])

    def graph_from_board(self):
        self.board.node_map = dict()
        self.board.wp_map = dict()
        self.graph = Graph(directed=False)
        self.graph.gp["h"] = self.graph.new_graph_property("long")
        self.graph.gp["b"] = self.graph.new_graph_property("bool")
        self.graph.gp["b"] = True if self.board.onturn=="b" else False
        owner_prop = self.graph.new_vertex_property("short")
        self.graph.vp.o = owner_prop
        filt_prop = self.graph.new_vertex_property("bool")
        self.graph.vp.f = filt_prop
        added_verts = dict()
        for i,wsn in enumerate(list(self.board.winsquarenums)):
            owner = self.owner_rev["f"]
            add_verts = []
            for ws in wsn:
                if self.board.position[ws] == "f":
                    add_verts.append(ws)
                elif self.board.position[ws] != self.owner_map[owner]:
                    if self.owner_map[owner] == "f":
                        owner = self.owner_rev[self.board.position[ws]]
                    else:
                        break
            else:
                ws_vert = self.graph.add_vertex()
                self.board.wp_map[int(ws_vert)] = wsn
                self.graph.vp.o[ws_vert] = owner
                for av in add_verts:
                    if av in added_verts:
                        my_v = added_verts[av]
                    else:
                        my_v = self.graph.add_vertex()
                        self.board.node_map[int(my_v)] = av
                        self.graph.vp.o[my_v] = 0
                        added_verts[av] = my_v
                    self.graph.add_edge(ws_vert,my_v)
        self.graph.vp.f.a = np.ones(self.graph.num_vertices())
        self.view = GraphView(self.graph,self.graph.vp.f)
        self.board.inv_maps()

    def get_actions(self,filter_superseeded=True,none_for_win=True):
        actions = []
        for node in self.view.vertices():
            if self.view.vp.o[node]!=0:
                continue
            left_to_own = 0
            go_there = False
            neigh_indices = set()
            for target in node.all_neighbors():
                neigh_indices.add(int(target))
                count = target.out_degree()
                if count==1:
                    if none_for_win and self.owner_map[self.view.vp.o[target]] == self.onturn:
                        return None
                    go_there = True
                left_to_own += count
            deg = node.out_degree()
            actions.append((-10000*int(go_there)-deg+left_to_own/deg,int(node),neigh_indices))
        actions.sort()
        if filter_superseeded:
        # Remove superseeded actions
            for i in range(len(actions)-1,-1,-1):
                for j in range(i-1,-1,-1):
                    if actions[i][2].issubset(actions[j][2]):
                        del actions[i]
                        break
        return [x[1] for x in actions]
    
    def make_move(self,square_node):
        win=False
        if type(square_node) == int:
            square_node = self.view.vertex(square_node)
        del_nodes = [square_node]
        lost_neighbors = defaultdict(int)
        for wp_node in square_node.all_neighbors():
            owner = self.owner_map[self.view.vp.o[wp_node]]
            if owner == "f":
                self.view.vp.o[wp_node] = self.owner_rev[self.onturn]
            elif owner != self.onturn:
                for sq_node in wp_node.all_neighbors():
                    i = self.view.vertex_index[sq_node]
                    if sq_node.out_degree() - lost_neighbors[i] == 1:
                        del_nodes.append(sq_node)
                    lost_neighbors[i]+=1
                del_nodes.append(wp_node)
            if wp_node.out_degree() == 1 and owner in ("f",self.onturn):
                win=True
            
        for del_node in del_nodes:
            self.view.vp.f[del_node] = False
        self.view.gp["b"] = not self.view.gp["b"]
        return win

    def negate_onturn(self,onturn):
        return "b" if onturn=="w" else ("w" if onturn=="b" else onturn)
        
    def threat_search(self,last_gain=None,last_cost=None,known_threats=None,gain=None,cost=None):
        if known_threats is None:
            known_threats=dict()
        if gain is None:
            gain = set()
        if cost is None:
            cost = set()
        if gain in self.known_gain_sets:
            return set(),False,[]
        legit_defenses = set()
        movelines = []
        winlines = []
        force_me_to = None
        vert_inds = dict()
        double_threat = dict()
        done = False
        if last_gain is None:
            self.known_gain_sets = []
            for vert in self.view.vertices():
                deg = vert.out_degree()
                owner = self.owner_map[self.view.vp.o[vert]]
                if owner != None:
                    if owner == self.onturn or owner=="f":
                        if deg == 1:
                            sq, = vert.all_neighbors()
                            ind = int(sq)
                            use_defenses = set()
                            use_defenses.add(ind)
                            winlines.append(use_defenses)
                            movelines.append([last_gain,ind,"deg1"])
                            done = True
                            break
                        elif deg == 2:
                            nod1,nod2 = vert.all_neighbors()
                            ind1,ind2 = int(nod1),int(nod2)
                            if ind1 in vert_inds and ind2!=vert_inds[ind1]:
                                double_threat[ind1]=(ind2,vert_inds[ind1])
                            if ind2 in vert_inds and ind1!=vert_inds[ind2]:
                                double_threat[ind2] = (ind1,vert_inds[ind2])
                            vert_inds[ind1] = ind2
                            vert_inds[ind2] = ind1
                    else:
                        if deg == 1:
                            sq, = vert.all_neighbors()
                            ind = int(sq)
                            if force_me_to is None:
                                force_me_to = ind
                                legit_defenses.add(ind)
                            else:
                                if ind != force_me_to:
                                    done = True
                        elif deg == 2:
                            sq1,sq2 = vert.all_neighbors()
                            legit_defenses.add(int(sq1))
                            legit_defenses.add(int(sq2))
        else:
            self.known_gain_sets.append(gain)
            rest_squares = set()
            for wp_ind in self.view.get_all_neighbors(last_cost):
                vert = self.view.vertex(wp_ind)
                if self.owner_map[self.view.vp.o[vert]] == self.onturn:
                    continue
                frees = set()
                for sq_ind in self.view.get_all_neighbors(wp_ind):
                    if sq_ind in gain:
                        break
                    if sq_ind not in cost:
                        frees.add(sq_ind)
                else:
                    if len(frees)==1:
                        ind, = frees
                        legit_defenses.add(ind)
                        if force_me_to is None:
                            force_me_to = ind
                        else:
                            if ind != force_me_to:
                                done = True
                    elif len(frees)==2:
                        for f in frees:
                            legit_defenses.add(f)
            for wp_ind in self.view.get_all_neighbors(last_gain):
                vert = self.view.vertex(wp_ind)
                if self.owner_map[self.view.vp.o[vert]] == self.negate_onturn(self.onturn):
                    continue
                frees = set()
                for sq_ind in self.view.get_all_neighbors(wp_ind):
                    if sq_ind in cost:
                        break
                    if sq_ind not in gain:
                        frees.add(sq_ind)
                else:
                    if len(frees) == 1:
                        sq, = frees
                        use_defenses = set()
                        use_defenses.add(sq)
                        winlines.append(use_defenses)
                        movelines.append([last_gain,sq,"deg1_inner"])
                        done = True
                    elif len(frees) == 2:
                        ind1,ind2 = frees
                        if ind1 in vert_inds and ind2!=vert_inds[ind1]:
                            double_threat[ind1]=(ind2,vert_inds[ind1])
                        if ind2 in vert_inds and ind1!=vert_inds[ind2]:
                            double_threat[ind2]=(ind1,vert_inds[ind2])
                        vert_inds[ind1] = ind2
                        vert_inds[ind2] = ind1
                    else:
                        rest_squares.update(frees)
            for key in known_threats:
                if key in rest_squares and not key in cost and not key in gain and not known_threats[key] in cost:
                    vert_inds[key] = known_threats[key]
        if not done:
            if force_me_to is not None:
                if force_me_to in vert_inds:
                    vert_inds = {force_me_to:vert_inds[force_me_to]}
                else:
                    done = True
        if not done:
            for n in set(vert_inds).intersection(set(double_threat)):
                use_defenses = legit_defenses.copy()
                use_defenses.add(double_threat[n][0])
                use_defenses.add(double_threat[n][1])
                use_defenses.add(n)
                winlines.append(use_defenses)
                movelines.append([last_gain,n,"double_threat"])
                done = True
        if not done:
            known_threats.update(vert_inds)
            for i,ind in enumerate(vert_inds.keys()):
                use_cost = cost.copy()
                use_gain = gain.copy()
                use_cost.add(vert_inds[ind])
                use_gain.add(ind)
                th_copy = known_threats.copy()
                del th_copy[ind]
                under_defs,win_here,move_here = self.threat_search(last_gain=ind,last_cost=vert_inds[ind],gain=use_gain,cost=use_cost,known_threats=th_copy)
                if win_here:
                    for move in move_here:
                        movelines.append([last_gain]+move)
                    under_defs.update(legit_defenses)
                    under_defs.add(ind)
                    under_defs.add(vert_inds[ind])
                    winlines.append(under_defs)
        if len(winlines)>0:
            return reduce(lambda x,y:x.intersection(y),winlines), True, movelines
        else:
            return set(), False, []

    def win_threat_search(self,one_is_enough=False,until_time=None):
        if until_time is not None and time.time() > until_time:
            return set()
        force_me_to = None
        vert_inds = dict()
        double_threat = dict()
        winmoves = set()
        loss = False
        for vert in self.view.vertices():
            deg = vert.out_degree()
            owner = self.owner_map[self.view.vp.o[vert]]
            if owner != None:
                if owner == self.onturn or owner=="f":
                    if deg == 1:
                        sq, = vert.all_neighbors()
                        winmoves.add(int(sq))
                        if one_is_enough:
                            return winmoves
                    elif deg == 2:
                        nod1,nod2 = vert.all_neighbors()
                        ind1,ind2 = int(nod1),int(nod2)
                        if ind1 in vert_inds and ind2!=vert_inds[ind1]:
                            double_threat[ind1]=(ind2,vert_inds[ind1])
                        if ind2 in vert_inds and ind1!=vert_inds[ind2]:
                            double_threat[ind2] = (ind1,vert_inds[ind2])
                        vert_inds[ind1] = ind2
                        vert_inds[ind2] = ind1
                else:
                    if deg == 1:
                        sq, = vert.all_neighbors()
                        ind = int(sq)
                        if force_me_to is None:
                            force_me_to = ind
                        else:
                            if ind != force_me_to:
                                loss = True
        if loss:
            return winmoves
        if force_me_to is not None:
            if force_me_to in vert_inds:
                vert_inds = {force_me_to:vert_inds[force_me_to]}
            else:
                return winmoves
        for n in set(vert_inds).intersection(set(double_threat)):
            winmoves.add(n)
            if one_is_enough:
                return winmoves
        if len(winmoves)>1 and one_is_enough:
            return winmoves
        storage = self.extract_storage()
        for ind in vert_inds:
            if ind in winmoves:
                continue
            self.make_move(ind)
            self.make_move(vert_inds[ind])
            if len(self.win_threat_search(one_is_enough=True,until_time=until_time)) > 0:
                winmoves.add(ind)
                if one_is_enough:
                    return winmoves
            self.load_storage(storage)
        return winmoves

    def draw_me(self,index=0):
        if self.view.num_vertices()==0:
            print("WARNING: Trying to draw graph without vertices")
            return
        fill_color = self.view.new_vertex_property("vector<float>")
        for vertex in self.view.vertices():
            x = self.view.vp.o[vertex]
            fill_color[vertex] = (0,0,1,1) if x==0 else ((0,1,1,1) if x==1 else ((0,0,0,1) if x==2 else (255,0,0,1)))
        graph_draw(self.view, vprops={"fill_color":fill_color}, vertex_text=self.view.vertex_index, output=f"game_state_{index}.pdf")

    def __str__(self):
        return self.name