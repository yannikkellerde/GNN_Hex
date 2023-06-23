"""An abstract class for a grid board game with an associated graph representation"""

from graph_tool.all import VertexPropertyMap, Vertex, Graph, GraphView
from typing import Set,List,Dict,FrozenSet
from abc import ABC, abstractmethod

class Abstract_board_game(ABC):
    game:"Abstract_graph_game"
    squares:int
    onturn:str

    @abstractmethod 
    def graph_from_board(self):
        """Construct the graph from the current board representation"""
        raise NotImplementedError


    def make_move(self, move:int):
        """Make a move on the board representation and update the graph representation.
        
        Args:
            move: The square the move is to be made on.
        """
        raise NotImplementedError
        

    def set_position(self,pos:List[str],onturn:str):
        """Set the board and graph representation to a given position.
        
        Args:
            pos: A list of strings with one string for each square.
            onturn: "w" or "b" depending on whose turn it is.
        """
        self.position = pos
        self.onturn = onturn
        self.graph_from_board()

    @abstractmethod
    def draw_me(self,pos=None) -> str:
        """Draw the board in the terminal"""
        raise NotImplementedError

