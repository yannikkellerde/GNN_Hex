from graph_game.abstract_graph_game import Abstract_graph_game
from graph_game.utils import is_fully_connected
from graph_tool.all import VertexPropertyMap, Graph, GraphView,graph_draw,Vertex,dfs_iterator,adjacency,shortest_distance
from typing import Union, List, Iterator, Set, Callable
import numpy as np
import scipy.linalg
import sklearn.preprocessing

class Clique_node_switching_game(Abstract_graph_game):
    terminals:List[Vertex]
    board_callback:Callable

    @property
    def onturn(self):
        return "m" if self.view.gp["m"] else "b" # m for maker, b for breaker

    def get_actions(self):
        return self.view.vertex_index.copy().fa[2:][self.view.vp.s.fa] # We assume terminals in vertex index 0 and 1 for efficiency here

    def dead_neighbor_removal(self,consider_set:List[int],iterate=True):
        """Remove cliques and squares with one or less neighbors"""
        if type(consider_set) not in (list,np.ndarray):
            consider_set = list(consider_set)
        next_set = {}
        for node,deg in zip(consider_set,self.view.get_total_degrees(consider_set)):
            if node<2:  # Terminals
                continue
            if deg<2:
                if self.view.vp.s[node]:
                    self.make_move(node,force_color="b")
                else:
                    if iterate:
                        next_set.update(set(self.view.get_all_neighbors(node)))
                    self.view.vp.f[node] = False
        if iterate and len(next_set)>0:
            next_next_set = self.dead_neighbor_removal(next_set,iterate=True)
            if self.view.vp.s[next(iter(next_set))]:
                next_set.update(next_next_set)
                return next_set
            return next_next_set

    def dead_and_captured(self,consider_set:Union[None,List[int],Set[int]]=None,iterate=True): 
        """Find dead and captured vertices and handle them appropriately

        Dead vertices and breaker captured vertices are removed. Maker captured vertices
        are removed and neighbors get connected. Uses local graph patterns to find captured
        and dead vertices.

        Args:
            consider_set: If not None, only check this subset of vertices
            iterate: iterate to check if new more nodes ended up captured or dead as a 
                     consequence of changes from last action on dead or captured cells
        """
        if consider_set is None:
            consider_set = self.view.vertices()
        more_to_consider = self.dead_neighbor_removal(consider_set,iterate=True) # Handle dead nodes
        consider_set = set(consider_set)
        consider_set.update(more_to_consider)
        neighborsets = dict()
        already = set()
        next_to_consider = set()
        for clique in consider_set:
            for neigh in self.view.iter_all_neighbors(clique):
                if neigh in already:
                    continue
                else:
                    already.add(neigh)
                neighset = frozenset(self.view.get_all_neighbors(neigh))
                if len(neighset)==2:
                    if neighset in neighborsets: # Maker captures
                        self.make_move(neighborsets[neighset],force_color="b",remove_dead_and_captured=False,raise_error=False)
                        self.make_move(neigh,force_color="m",remove_dead_and_captured=False,raise_error=False)
                        next_to_consider.update(neighset)
                    else:
                        for next_clique in neighset:
                            if self.view.vertex(next_clique).out_degree() == 2:
                                for deep_neigh in self.view.iter_all_neighbors(next_clique):
                                    if self.view.vertex(deep_neigh).out_degree()==2 and deep_neigh!=neigh: # Breaker captures
                                        self.make_move(neigh,force_color="b",remove_dead_and_captured=False,raise_error=False)
                                        self.make_move(deep_neigh,force_color="b",remove_dead_and_captured=False,raise_error=False)
                                        next_to_consider.update(neighset)
                                        next_to_consider.update(set(self.view.get_all_neighbors(deep_neigh)))
                    neighborsets[neighset] = neigh
        if iterate:
            self.dead_and_captured(next_to_consider,iterate=True)

                    
    def make_move(self,square_node:Union[int,Vertex],force_color=None,remove_dead_and_captured=False,raise_error=True):
        """Make a move by choosing a vertex in the graph

        Args:
            square_node: Vertex or vertex index to move to
            force_color: 'm' or 'b', if set, play for this player instead of who is on-turn.
            remove_dead_and_captured: If true, remove/fill any noded that became dead/captured as
                                      a consequence of this move
        """
        if force_color is None:
            makerturn = self.view.gp["m"]
        else:
            makerturn = force_color=="m"
        if type(square_node)==int:
            square_node = self.view.vertex(square_node)
        if not self.view.vp.s[square_node]:
            if raise_error:
                raise ValueError(f"Invalid Vertex {int(square_node)}, can't select clique vertex")
            else:
                return
        if not self.view.vp.f[square_node]:
            if raise_error:
                raise ValueError(f"Invalid Vertex {int(square_node)}, vertex has already been removed")
            else:
                return
        if square_node in self.terminals:
            if raise_error:
                raise ValueError(f"Can't play at terminal node {int(square_node)}")
            else:
                return
        neigh_cliques = self.view.get_all_neighbors(square_node)
        if makerturn:
            keep = neigh_cliques[0]
            for neigh_clique in neigh_cliques[1:]:
                for neigh_square in self.view.iter_all_neighbors(neigh_clique):
                    self.view.edge(neigh_square,keep,add_missing=True)
                self.view.vp.f[neigh_clique] = False
        self.view.vp.f[square_node] = False
        if force_color is None:
            self.view.gp["m"] = not self.view.gp["m"]
        if self.board_callback is not None:
            self.board_callback(int(square_node),makerturn)
        if remove_dead_and_captured:
            self.dead_and_captured(neigh_cliques,True)
        else:
            self.dead_neighbor_removal([keep] if makerturn else neigh_cliques,False)

    def who_won(self):
        if shortest_distance(self.view,self.terminals[0],self.terminals[1])<=2:
            return "m"
        for e in dfs_iterator(self.view,self.terminals[0]):
            if e.target() == self.terminals[1]:
                return None
        return "b"

    def move_wins(self,move_vertex:Union[Vertex,int]) -> bool:
        if type(move_vertex) == int:
            move_vertex = self.view.vertex(move_vertex)
        if self.view.gp["m"]:
            found0 = False
            found1 = False
            for neigh in self.view.iter_all_neighbors(move_vertex):
                if self.view.edge(neigh,self.terminals[0]):
                    found0 = True
                if self.view.edge(neigh,self.terminals[1]):
                    found1 = True
            return found0 and found1
        else:
            self.view.vp.f[move_vertex] = False
            for e in dfs_iterator(self.view,self.terminals[0]):
                if e.target() == self.terminals[1]:
                    self.view.vp.f[move_vertex] = True
                    return False
            self.view.vp.f[move_vertex] = True
            return True

    @staticmethod
    def from_graph(graph:Graph):
        g = Clique_node_switching_game()
        g.graph = graph
        g.view = GraphView(g.graph,vfilt=g.graph.vp.f)
        g.board = None
        g.name = "Clique_node_switching_game"
        return g

    def prune_irrelevant_subgraphs(self) -> bool:
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
            self.make_move(vi,force_color="b")
        return breaker_wins

    def draw_me(self,fname="clique_switching_game.pdf",vprop=None):
        """Draw the state of the graph and save it into a pdf file.

        Args:
            fname: Filename to save to
            vprop: Optional vertex property map of type int or double to display
        """
        fill_color = self.view.new_vertex_property("vector<float>")
        shape = self.view.new_vertex_property("string")
        size = self.view.new_vertex_property("int")
        for vertex in self.view.vertices():
            if vertex in self.terminals:
                shape[vertex] = "circle"
                fill_color[vertex] = (1,0,0,1)
                size[vertex] = 25
            else:
                if self.view.vp.s[vertex]:
                    shape[vertex] = "hexagon"
                    fill_color[vertex] = (0,0,0,1)
                    size[vertex] = 15
                else:
                    shape[vertex] = "square"
                    fill_color[vertex] = (0,0,1,1)
                    size[vertex] = 15
        vprops = {"fill_color":fill_color,"shape":shape,"size":size}
        graph_draw(self.view, vprops=vprops, vertex_text=vprop, output=fname)
