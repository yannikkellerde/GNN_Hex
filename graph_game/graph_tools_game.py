from __future__ import annotations
import math
from copy import copy,deepcopy
from typing import NamedTuple, Union
from functools import reduce
from collections import defaultdict
import time
import numpy as np
import pickle
import time
from graph_game.graph_board_game import Board_game
from graph_tool.all import *
from graph_tools_hashing import wl_hash
from utils.general_utils import hasattr 

class Graph_Store(NamedTuple):
    """A minimal storage NamedTuple that contains all information to reconstruct a game state if
    the basic graph structure is known."""
    owner_map:VertexPropertyMap
    filter_map:VertexPropertyMap
    blackturn:bool

class Graph_game():
    graph: Graph
    view: GraphView
    name:str
    board:Union[None,Board_game]
    def __init__(self):
        self.owner_map = {0:None,1:"f",2:"b",3:"w"}
        self.owner_rev = {val:key for key,val in self.owner_map.items()}
        self.known_gain_sets = []
        self.psets = {"bp":set(),"bd":set(),"wp":set(),"wd":set()}

    @staticmethod
    def from_graph(graph) -> Graph_game:
        """Create a game from a graph-tool graph.

        Args:
            graph: A graph-tool graph.
        Returns:
            A graph-tool game
        """
        g = Graph_game()
        g.graph = graph
        g.view = GraphView(g.graph,g.graph.vp.f)
        g.board = None
        g.name = "Graph_game"
        return g

    @property
    def hash(self):
        return self.view.gp["h"]

    @property
    def onturn(self):
        return "b" if self.view.gp["b"] else "w"

    def hashme(self):
        """Compute the Weisfeiler-Lehmann hash from the current graph and store it as a
        global property of the graph. (self.view.gp['h'])
        """
        wl_hash(self.view,self.view.vp.o,iterations=3)

    def load_storage(self,storage:Graph_Store):
        """Load the game state from it's property maps.

        Args:
            storage: A Graph_Store Namedtuple with owner map, filter map and
                     information about who's turn it is.
        """
        self.graph.vp.f = storage.filter_map.copy()
        self.view = GraphView(self.graph,vfilt=self.graph.vp.f)
        self.view.vp.o = storage.owner_map.copy()
        self.view.gp["b"] = storage.blackturn

    def extract_storage(self) -> Graph_Store:
        """Extract a minimal storage object that contains all information
        required to reconstruct the current game state if the basic graph structure is known.

        Returns:
            A Graph_Store Namedtuple with the owner and filter property maps of the graph
            as well as blackturn signaling the color of the player who is onturn.
        """
        return Graph_Store(owner_map = self.view.vp.o.copy(),
                           filter_map = self.view.vp.f.copy(),
                           blackturn = self.view.gp["b"])

    def graph_from_board(self):
        """ Construct the game graph from a board representation of the game.

        Assumes that self.board is properly with winsquarenums (winpatterns in board representation)
        and position, a 2D list that represents the board position.
        Creates graph-tools game graph and adds vetices for squares and winpatterns. Adds edges between
        them.
        Created board.node_map, board.wp_map and board.inv_map, that map the winpatterns/squares in the
        graph representation to the location in the board representation.
        """
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
        """Find and sort all moves that are possible in the current game state.

        The possible moves are the indices of all square vertices. The moves are
        sorted ascending by the following heuristic:
        -degree_of_vertex+sum_of_degrees_of_connected_winpatterns/degree_of_vertex
        Moves that are connected to a winpattern with degree one are put to the front.

        Args:
            filter_superseeded: Remove all moves that correspond to a subset of the winpatters
                                connected to another move.
            none_for_win: If there is a move that instantly wins the game, just return None instead
                          of the actions.
        
        Returns:
            A sorted list of the indices corresponding to possible moves.
        """
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
    
    def make_move(self,square_node:Union[int,Vertex]) -> bool:
        """Execute a move on the game graph

        Removes the selected vertex from the graph. Checks all winpattern
        nodes connected to the removed vertex.
        A) If the winpattern node is uncolored, color it in the moving players color
        B) If the winpattern node is in the moving players color and the node has no
           neighbors left, return True, signaling a player win.
        C) If the winpattern node is in the opponents color, remove the winpattern.
           search all square vertices connected to the removed winpattern. Remove them
           if their degree is 0.
        
        Args:
            square_node: Either a graph square vertex on the index of a graph square vertex
        Returns:
            win: Did the player win the game with his current move?
        """
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

    def check_move_val(self,moves,priorize_sets=True):
        """{"-4":"White wins (Forced Move)","-3":"White wins (Threat search)","-2":"White wins (Proofset)",
         "-1":"White wins or draw","u":"Unknown",0:"Draw",1:"Black wins or draw",2:"Black wins (Proofset)",
         3:"Black wins (Threat search)",4:"Black wins (Forced Move)"}"""
        winmoves = self.win_threat_search(one_is_enough=False,until_time=time.time()+5)
        self.view.gp["b"] = not self.view.gp["b"]
        defense_vertices,has_threat,_ = self.threat_search()
        self.view.gp["b"] = not self.view.gp["b"]
        results = []
        storage = self.extract_storage()
        for move in moves:
            val = "u"
            self.load_storage(storage)
            if has_threat and move not in defense_vertices and move not in winmoves:
                if self.onturn=="b":
                    val = -3
                else:
                    val = 3
            else:
                self.make_move(move)
                self.hashme()
                if self.hash in self.psets["wp"]:
                    val = -2
                elif self.hash in self.psets["bp"]:
                    val = 2
                elif self.hash in self.psets["wd"]:
                    val = 1
                if self.hash in self.psets["bd"]:
                    if val == 1:
                        val = 0
                    elif val =="u":
                        val = -1
                if val=="u" or not priorize_sets:
                    if self.view.num_vertices() == 0:
                        val = 0
                    else:
                        if move in winmoves:
                            if self.onturn=="b":
                                val = -4
                            else:
                                val = 4
                        else:
                            movs = len(self.win_threat_search(one_is_enough=True,until_time=time.time()+0.5))>0
                            if movs:
                                if self.onturn=="b":
                                    val = 4
                                else:
                                    val = -4
            results.append(val)
        return results

    def negate_onturn(self,onturn):
        return "b" if onturn=="w" else ("w" if onturn=="b" else onturn)
        
    def threat_search(self,last_gain=None,last_cost=None,known_threats=None,gain=None,cost=None):
        """Implements a threat-space search for games on graphs. Inspired by Go-Moku and Threat-Space Search (1994).

        Threats are defined as squares connected to a winpattern node of degree 2 that is colored in the moving players color
        or is uncolored. These are searched in a breadth-first way, assuming that the opponent always defends the threats until
        it finds a forced sequence that leads to a win (e.g. with a double-threat).
        All forced wins found are guaranteed to be correct. However, it is not guaranteed that this search will find all forced ways
        to victory.
        Arguments are for recursion purposes. Call without parameters.

        """
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
        """Draw the state of the graph and save it into a pdf file.

        Args:
            index: An index to append to the name of the pdf file.
        """
        if self.view.num_vertices()==0:
            print("WARNING: Trying to draw graph without vertices")
            return
        fill_color = self.view.new_vertex_property("vector<float>")
        for vertex in self.view.vertices():
            x = self.view.vp.o[vertex]
            fill_color[vertex] = (0,0,1,1) if x==0 else ((0,1,1,1) if x==1 else ((1,0,0,1) if x==2 else (0,0,0,1)))
        vprops = {"fill_color":fill_color}
        if hasattr(self.graph.vp,"w"):
            stroke_color = self.view.new_vertex_property("vector<float>")
            for vertex in self.view.vertices():
                if self.view.vp.o[vertex] != 0:
                    stroke_color[vertex] = (0,0,0,0)
                else:
                    stroke_color[vertex] = (1,0,0,1) if self.view.vp.w[vertex][0] else ((0,0,0,1) if self.view.vp.w[vertex][1] else (0,0,1,1))
            vprops["color"] = stroke_color
        graph_draw(self.view, vprops=vprops, vertex_text=self.view.vertex_index, output=f"game_state_{index}.pdf")

    def __str__(self) -> str:
        return self.name