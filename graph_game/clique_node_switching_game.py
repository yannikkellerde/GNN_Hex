from graph_game.abstract_graph_game import Abstract_graph_game
from graph_game.utils import is_fully_connected
from graph_tool.all import VertexPropertyMap, Graph, GraphView,graph_draw,Vertex,dfs_iterator,adjacency
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
        next_set = {}
        for node,deg in zip(consider_set,self.view.get_total_degrees(consider_set)):
            if deg<2:
                if self.view.vp.s[node]:
                    self.make_move(node,force_color="b")
                else:
                    if iterate:
                        next_set.update(set(self.view.get_all_neighbors(node)))
                    self.view.vp.f[node] = False
        if iterate and len(next_set)>0:
            self.dead_neighbor_removal(list(next_set))



    def make_move(self,square_node:Union[int,Vertex],force_color=None,remove_dead_and_captured=False):
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
            raise ValueError(f"Invalid Vertex {int(square_node)}, can't select clique vertex")
        neigh_cliques = self.view.get_all_neighbors(square_node)
        if makerturn:
            keep = neigh_cliques[0]
            for neigh_clique in neigh_cliques[1:]:
                for neigh_square in self.view.iter_all_neighbors(neigh_clique):
                    self.view.edge(neigh_square,keep,add_missing=True)
                self.view.vp.f[neigh_clique] = False
        self.view.vp.f[square_node] = False
        self.dead_neighbor_removal(neigh_cliques)
        if force_color is None:
            self.view.gp["m"] = not self.view.gp["m"]
        if self.board_callback is not None:
            self.board_callback(int(square_node),makerturn)
        if remove_dead_and_captured:
            self.dead_and_captured(self.view.get_out_neighbors(square_node),True)
