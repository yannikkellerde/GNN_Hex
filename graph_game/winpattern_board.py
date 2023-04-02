"""A winpattern game as a 2d grid. Winpattern games include games such as Tic-Tac-Toe, Gomoku or Qango"""

from graph_tool.all import VertexPropertyMap, Vertex, Graph, GraphView
import pickle
import sys,os
import math
import numpy as np
import time
from blessings import Terminal
from typing import Set,List,Dict,FrozenSet


class Winpattern_board():
    """2d Grid representation of a two player board game.

    Manages the 2d grid representation of a graph game. Is usually used in conjunction with
    a Winpattern_game. Winpattern_board is subclassed in graph_game/graph_tools_games.py.

    Attributes:
        game: The corresponding Winpattern_game object.
        winsquarenums: A set of frozensets. Each frozenset is a set of squares in a winpattern.
        position: A list of strings with one string for each square. The string is either "b" for black,
                  "w" for white, "f" for free/empty, or "U" for unknown.
        squares: Number of squares in the game. (Assumed to be a square number.)
        onturn: "w" or "b" depending on whose turn it is.
        wp_map: A dictionary mapping a winpattern to the set of corresponding squares.
        wp_map_rev: The inverted wp_map
        node_map: A dictionary mapping a node in the graph to the corresponding square number on the board.
        node_map_rev: The inverted node_map
        rulesets: An optional dictionary containing sets of nodes that are blocked by a special rule on the first turns.
    
    Methods:
        inv_maps: computes wp_map_rev and node_map_rev.
        pos_from_graph: converts the game.graph back into the board game position
        make_move: Make a move on the board representation.
        set_position: Set the board and graph representation to a given position.
        draw_me_with_prediction: print the board state into the terminal with colored indicators for the prediction.
        draw_me: print the board state into the terminal.
        get_blocked_squares: return a set of squares that are blocked by a special rule on the first turns.
    """
    game:"Winpattern_game"
    winsquarenums:Set[FrozenSet[int]]
    position:List[str]
    squares:int
    onturn:str
    wp_map:Dict[int,FrozenSet[int]]
    node_map:Dict[int,int]
    wp_map_rev:Dict[FrozenSet[int],int]
    node_map_rev:Dict[int,int]
    rulesets:Dict[str,Set[int]]

    def __init__(self):
        self.onturn = "b"
        self.node_map = dict()
        self.wp_map = dict()
    
    def inv_maps(self):
        """Computes wp_map_rev and node_map_rev."""
        self.wp_map_rev = {value:key for key,value in self.wp_map.items()}
        self.node_map_rev = {value:key for key,value in self.node_map.items()}

    def pos_from_graph(self) -> List[str]:
        """Converts the game.graph back into the 2d grid position.
        
        Returns:
            A list of strings with one string for each square. The string is either "b" for black,
            "w" for white, "f" for free/empty, or "U" for unknown.
        """
        known_nods = [self.node_map[int(x)] for x in self.game.view.vertices() if self.game.owner_map[self.game.view.vp.o[x]] is None]
        pos = ["U"]*self.squares
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


    def graph_from_board(self):
        """ Construct the game graph from a board representation of the game.

        Assumes that the board is properly with winsquarenums (winpatterns in board representation)
        and position, a 2D list that represents the board position.
        Creates graph-tools game graph and adds vetices for squares and winpatterns. Adds edges between
        them.
        Created board.node_map, board.wp_map and board.inv_map, that map the winpatterns/squares in the
        graph representation to the location in the board representation.
        """
        self.node_map = dict()
        self.wp_map = dict()
        self.game.graph = Graph(directed=False)
        self.game.graph.gp["h"] = self.game.graph.new_graph_property("long")
        self.game.graph.gp["b"] = self.game.graph.new_graph_property("bool")
        self.game.graph.gp["b"] = True if self.onturn=="b" else False
        owner_prop = self.game.graph.new_vertex_property("short")
        self.game.graph.vp.o = owner_prop
        filt_prop = self.game.graph.new_vertex_property("bool")
        self.game.graph.vp.f = filt_prop
        added_verts = dict()
        for wsn in list(self.winsquarenums):
            owner = self.game.owner_rev["f"]
            add_verts = []
            for ws in wsn:
                if self.position[ws] == "f":
                    add_verts.append(ws)
                elif self.position[ws] != self.game.owner_map[owner]:
                    if self.game.owner_map[owner] == "f":
                        owner = self.game.owner_rev[self.position[ws]]
                    else:
                        break
            else:
                ws_vert = self.game.graph.add_vertex()
                self.wp_map[int(ws_vert)] = wsn
                self.game.graph.vp.o[ws_vert] = owner
                for av in add_verts:
                    if av in added_verts:
                        my_v = added_verts[av]
                    else:
                        my_v = self.game.graph.add_vertex()
                        self.node_map[int(my_v)] = av
                        self.game.graph.vp.o[my_v] = 0
                        added_verts[av] = my_v
                    self.game.graph.add_edge(ws_vert,my_v)
        self.game.graph.vp.f.a = np.ones(self.game.graph.num_vertices())
        self.game.view = GraphView(self.game.graph,self.game.graph.vp.f)
        self.inv_maps()

    def make_move(self, move:int):
        """Make a move on the board representation and update the graph representation.
        
        Args:
            move: The square the move is to be made on."""
        self.position[move] = self.onturn
        self.onturn = "b" if self.onturn == "w" else "b"
        self.graph_from_board()      

    def set_position(self,pos:List[str],onturn:str):
        """Set the board and graph representation to a given position.
        
        Args:
            pos: A list of strings with one string for each square.
            onturn: "w" or "b" depending on whose turn it is.
        """
        self.position = pos
        self.onturn = onturn
        self.graph_from_board()

    def draw_me_with_prediction(self,vprop:VertexPropertyMap,numbered=True) -> str:
        """Print the board state into the terminal with colored indicators for the prediction.
        
        Args:
            vprop: The vertex property map containing the prediction.
        Returns:
            A string with the printed board.
        """
        letters = "abcdefghijklmnopqrstuvwxyz"
        t = Terminal()
        root = int(math.sqrt(self.squares))
        out_str=""
        if numbered:
            out_str += "  "+"".join(letters[i] for i in range(root))+" "+"\n"
        out_str += " "+"#"*(root+2)
        out_str+="\n"
        for row in range(root):
            if numbered:
                out_str+=str(row+1)
            else:
                out_str+=" "
            out_str+="#"
            for col in range(root):
                sq = col+row*root
                letter = " " if self.position[sq]=="f" else self.position[sq]
                if self.position[sq]=="f" and vprop[self.node_map_rev[sq]][0]:
                    out_str+=t.on_green(letter)
                elif self.position[sq]=="f" and vprop[self.node_map_rev[sq]][1]:
                    out_str+=t.on_red(letter)
                else:
                    out_str+=letter
            out_str+="#\n"
        out_str += " "+"#"*(root+2)
        #print(out_str)
        return out_str


    def draw_me(self,pos=None) -> str:
        """Print the board state into the terminal.
        
        Args:
            pos: The position to be printed. If None, the current position is printed.

        Returns:
            A string with the printed board.
        """
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

    def get_blocked_squares(self,ruleset:str) -> Set[int]:
        """Return a set of squares that are blocked by a special rule on the first turns.
        
        Args:
            ruleset: The name of the ruleset to consider
        
        Returns:
            A set of squares that are blocked by a special rule on the first turns."""
        self.inv_maps()
        blocked_moves = set(self.node_map_rev[x] for x in self.rulesets[ruleset])
        return blocked_moves
