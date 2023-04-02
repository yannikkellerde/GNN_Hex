"""An abstract class for a game played on a graph"""

from abc import ABC,abstractmethod
from graph_tool.all import VertexPropertyMap,Graph,GraphView,graph_draw,Vertex
from typing import Union,List

class Abstract_graph_game(ABC):
    graph:Graph
    view:GraphView
    name:str
    board:Union[None,"Abstract_board_game"]

    @staticmethod
    @abstractmethod
    def from_graph(graph:Graph):
        """Create graph game from a given graph-tools Graph"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def onturn(self):
        """ color of who's turn it is depending on who's turn it is."""
        raise NotImplementedError

    @abstractmethod
    def get_actions(self) -> List[int]:
        """List all possible moves in the current position"""
        raise NotImplementedError

    @abstractmethod
    def move_wins(self,move_node:Union[int,Vertex]) -> bool:
        """Check if a move wins the game

        Args:
            square_node: The vertex where the player who is on turn plays
        
        Returns:
            weather the move wins
        """
        raise NotImplementedError

    @abstractmethod
    def make_move(self,square_node:Union[int,Vertex]) -> bool:
        """Make a move in the graph game

        Args:
            square_node: index of vertex to select, or the vertex itself
        Returns:
            True if move is legal, otherwise False    
        """
        raise NotImplementedError

    @abstractmethod
    def draw_me(self):
        """Draw the state of the graph and save as pdf"""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name
