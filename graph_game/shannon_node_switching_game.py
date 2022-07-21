from graph_game.abstract_graph_game import Abstract_graph_game
from graph_tool.all import VertexPropertyMap, Graph, GraphView,graph_draw,Vertex
from typing import Union, List

class Node_switching_game(Abstract_graph_game):
    terminals:List[Vertex]

    def __init__(self):
        pass

    @property
    def onturn(self):
        pass

    def get_actions(self,filter_superseeded=True):
        pass

    def make_move(self,square_node:Union[int,Vertex]):
        pass


    @staticmethod
    def from_graph(graph:Graph):
        g = Node_switching_game()
        g.graph = graph
        g.view = GraphView(g.graph)
        g.board = None
        g.name = "Graph_game"
        return g

    def draw_me(self,fname="node_switching.pdf"):
        """Draw the state of the graph and save it into a pdf file.

        Args:
            index: An index to append to the name of the pdf file.
        """
        if self.view.num_vertices()==0:
            print("WARNING: Trying to draw graph without vertices")
            return
        fill_color = self.view.new_vertex_property("vector<float>")
        shape = self.view.new_vertex_property("string")
        size = self.view.new_vertex_property("int")
        for vertex in self.view.vertices():
            if vertex in self.terminals:
                shape[vertex] = "circle"
                fill_color[vertex] = (1,0,0,1)
                size[vertex] = 25
            else:
                shape[vertex] = "hexagon"
                fill_color[vertex] = (0,0,0,1)
                size[vertex] = 15
        vprops = {"fill_color":fill_color,"shape":shape,"size":size}
        graph_draw(self.view, vprops=vprops, vertex_text=self.view.vertex_index, output=fname)
