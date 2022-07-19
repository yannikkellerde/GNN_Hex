from graph_game.abstract_graph_game import Abstract_graph_game
from graph_tool.all import VertexPropertyMap, Graph, GraphView,graph_draw,Vertex
from typing import Union, List

class Node_switching_game(Abstract_graph_game):
    terminals:List[Vertex]

    def __init__(self):
        pass


    @staticmethod
    def from_graph(graph:Graph):
        g = Node_switching_game()
        g.graph = graph
        g.view = GraphView(g.graph)
        g.board = None
        g.name = "Graph_game"
        return g

