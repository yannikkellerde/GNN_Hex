from graph_game.abstract_graph_game import Abstract_graph_game
from graph_tool.all import VertexPropertyMap, Graph, GraphView,graph_draw,Vertex,dfs_iterator,adjacency
from typing import Union, List
import numpy as np
import scipy.linalg
import sklearn.preprocessing

class Node_switching_game(Abstract_graph_game):
    terminals:List[Vertex]

    def __init__(self):
        pass

    @property
    def onturn(self):
        return "m" if self.view.gp["m"] else "b" # m for maker, b for breaker

    def get_actions(self):
        return self.view.vertex_index.copy().fa[2:] # We assume terminals in vertex index 0 and 1 for efficiency here

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
        g.name = "Shannon_node_switching_game"
        return g

    def prune_irrelevant_subgraphs(self) -> True:
        """Prune all subgraphs that are not connected to any terminal nodes

        As a side effect this will find out if the position is won for breaker

        Returns:
            If the position is won for breaker
        """
        found_vertices = dfs_iterator(self.view,source=self.terminals[0],array=True)
        if int(self.terminals[1]) not in found_vertices:
            new_found_vertices = dfs_iterator(self.view,source=self.terminals[1],array=True)
            if len(found_vertices)>0:
                if len(new_found_vertices)>0:
                    found_vertices = np.concatenate((found_vertices,new_found_vertices))
            else:
                if len(new_found_vertices)>0:
                    found_vertices = new_found_vertices
                else:
                    found_vertices = np.array([0,1])
            breaker_wins = True
        breaker_wins = False
        leftovers = set(self.view.get_vertices())-set(found_vertices.flatten())-set((0,1))
        for vi in leftovers:
            self.view.vp.f[self.view.vertex(vi)] = False
        return breaker_wins

    def compute_node_voltages_exact(self):
        adj = adjacency(self.view).toarray()
        adj = sklearn.preprocessing.normalize(adj,norm="l1")
        i1 = int(self.terminals[0])
        i2 = int(self.terminals[1])
        adj[i1] = 0
        adj[i2] = 0
        adj -= np.eye(adj.shape[0])
        adj[i1,i1] = 1
        adj[i2,i2] = 1
        b = np.zeros(adj.shape[0])
        b[i2] = 100
        voltages = scipy.linalg.solve(adj,b,overwrite_a=True,overwrite_b=True)
        v_prop = self.view.new_vertex_property("double")
        v_prop.fa = voltages
        return v_prop

    def compute_voltage_drops(self,voltage_prop):
        d_prop = self.view.new_vertex_property("double")
        for vertex in self.view.vertices():
            if vertex not in self.terminals:
                my_volt = voltage_prop[vertex]
                num_lower_neighs = 0
                num_higher_neighs = 0
                lower_sum = 0
                higher_sum = 0
                for neighbor in vertex.out_neighbors():
                    n_volt = voltage_prop[neighbor]
                    if n_volt>my_volt:
                        num_higher_neighs+=1
                        higher_sum += n_volt
                    else:
                        num_lower_neighs+=1
                        lower_sum += n_volt
                drop_high = higher_sum-my_volt*num_higher_neighs
                drop_low = my_volt*num_lower_neighs-lower_sum
                if not np.isclose(drop_low,drop_high):
                    self.draw_me("error_graph.pdf",voltage_prop)
                assert np.isclose(drop_low,drop_high)
                d_prop[vertex] = drop_low
        return d_prop



    def draw_me(self,fname="node_switching.pdf",vprop=None):
        """Draw the state of the graph and save it into a pdf file.

        Args:
            fname: Filename to save to
            vprop: Optional vertex property map of type int or double to display
        """
        if vprop is None:
            vprop = self.view.vertex_index
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
        graph_draw(self.view, vprops=vprops, vertex_text=vprop, output=fname)
