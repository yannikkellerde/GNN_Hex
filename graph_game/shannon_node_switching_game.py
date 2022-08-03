from graph_game.abstract_graph_game import Abstract_graph_game
from graph_tool.all import VertexPropertyMap, Graph, GraphView,graph_draw,Vertex,dfs_iterator
from typing import Union, List

class Node_switching_game(Abstract_graph_game):
    terminals:List[Vertex]

    def __init__(self):
        pass

    @property
    def onturn(self):
        return "m" if self.view.gp["m"] else "b" # m for maker, b for breaker

    def get_actions(self):
        return self.view.vertex_index.copy().fa

    def make_move(self,square_node:Union[int,Vertex]):
        if type(square_node)==int:
            square_node = self.view.vertex(square_node)
        if self.view.gp["m"]:
            for vertex1 in self.view.iter_all_neighbors(square_node):
                for vertex2 in self.view.iter_all_neighbors(square_node):
                    if vertex1!=vertex2:
                        self.view.edge(vertex1,vertex2,add_missing=True)
        self.view.vp.f[square_node] = False
        self.view.gp["m"] = not self.view.gp["m"]

    def who_won(self):
        if self.view.edge(self.terminals[0],self.terminals[1]):
            return "m"
        for e in dfs_iterator(self.view,self.terminals[0]):
            if e.target() == self.terminals[1]:
                return None
        return "b"
    
    def move_wins(self,move_vertex:Union[Vertex,int]) -> bool:
        if type(move_vertex) == int:
            move_vertex = self.view.vertex(move_vertex)
        if self.view.gp["m"]:
            if self.view.edge(move_vertex,self.terminals[0]) and self.view.edge(move_vertex,self.terminals[1]):
                return True
        else:
            self.view.vp.f[move_vertex] = False
            for e in dfs_iterator(self.view,self.terminals[0]):
                if e.target() == self.terminals[1]:
                    self.view.vp.f[move_vertex] = True
                    return False
            self.view.vp.f[move_vertex] = True
            return True
        return False

    @staticmethod
    def from_graph(graph:Graph):
        g = Node_switching_game()
        g.graph = graph
        g.view = GraphView(g.graph)
        g.board = None
        g.name = "Winpattern_game"
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
