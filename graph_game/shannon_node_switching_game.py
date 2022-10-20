from graph_game.abstract_graph_game import Abstract_graph_game
from graph_game.utils import is_fully_connected, double_loop_iterator
from graph_tool.all import VertexPropertyMap, Graph, GraphView,graph_draw,Vertex,dfs_iterator,adjacency,boykov_kolmogorov_max_flow,min_st_cut,sfdp_layout
from typing import Union, List, Iterator, Set, Callable, Tuple
import numpy as np
from graph_game.utils import to_directed_graph
import scipy.linalg
import sklearn.preprocessing
from itertools import tee
import time
from dataclasses import dataclass
from graph_game.graph_tools_hashing import wl_hash

class Node_switching_game(Abstract_graph_game):
    terminals:List[Vertex]
    board_callback:Union[None,Callable]

    def __init__(self):
        self.board_callback = None
        self.total_num_moves = 0
        self.creation_time = time.perf_counter()
        self.response_set_maker = {}
        self.response_set_breaker = {}
        self.callback_everything = True

    @property
    def onturn(self):
        return "m" if self.view.gp["m"] else "b" # m for maker, b for breaker

    @property
    def not_onturn(self):
        return "b" if self.view.gp["m"] else "m" # m for maker, b for breaker

    def get_unique_starting_moves(self):
        assert self.board is not None
        return [self.board.board_index_to_vertex[m] for m in self.board.get_all_unique_starting_moves()]

    def get_actions(self):
        return self.view.vertex_index.copy().fa[2:] # We assume terminals in vertex index 0 and 1 for efficiency here

    def hashme(self):
        some_prop = self.view.new_vertex_property("bool")
        some_prop.a = 0
        some_prop[self.view.vertex(0)] = 1
        some_prop[self.view.vertex(1)] = 1
        return wl_hash(self.view,some_prop,self.view.gp["m"])

    def _fix_teminal_connections(self,terminal):
        change_set = set()
        for v1,v2 in double_loop_iterator(self.view.iter_all_neighbors(terminal)):
            edge = self.view.edge(v1,v2)
            if edge is not None:
                self.view.remove_edge(edge)
                change_set.add(v1)
                change_set.add(v2)
        return change_set

    def get_response(self,move,for_maker):
        move = int(move)
        response_vertex = None
        if for_maker and move in self.response_set_maker:
            response_vertex = self.response_set_maker[move]
        if not for_maker and move in self.response_set_breaker:
            response_vertex = self.response_set_breaker[move]

        if response_vertex is not None:
            return self.board.vertex_index_to_board_index[response_vertex]
        else:
            return None

    def make_move(self,square_node:Union[int,Vertex],force_color=None,remove_dead_and_captured=False):
        """Make a move by choosing a vertex in the graph

        Args:
            square_node: Vertex or vertex index to move to
            force_color: 'm' or 'b', if set, play for this player instead of who is on-turn.
            remove_dead_and_captured: If true, remove/fill any noded that became dead/captured as
                                      a consequence of this move
        """
        if force_color is None:
            self.total_num_moves+=1
            makerturn = self.view.gp["m"]
        else:
            makerturn = force_color=="m"
        if type(square_node)==int:
            square_node = self.view.vertex(square_node)
        change_set=set()
        if makerturn:
            have_to_fix = None
            for vertex1,vertex2 in double_loop_iterator(self.view.iter_all_neighbors(square_node)):
                if vertex1 in (0,1):
                    have_to_fix = vertex1
                if not ((self.view.edge(vertex1,self.terminals[0]) and self.view.edge(vertex2,self.terminals[0])) or 
                        (self.view.edge(vertex1,self.terminals[1]) and self.view.edge(vertex2,self.terminals[1]))):
                    self.view.edge(vertex1,vertex2,add_missing=True)
            if have_to_fix is not None:
                change_set = self._fix_teminal_connections(have_to_fix)
        self.view.vp.f[square_node] = False
        if force_color is None:
            self.view.gp["m"] = not self.view.gp["m"]
        if self.board_callback is not None and (self.callback_everything or force_color is None):
            self.board_callback(int(square_node),makerturn)
        if remove_dead_and_captured:
            self.dead_and_captured(set(self.view.iter_all_neighbors(square_node)).union(change_set),True)
        return change_set


    def dead_and_captured(self,consider_set:Union[None,List[int],Set[int]]=None,iterate=False):
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
            consider_set = self.view.get_vertices()[2:]
        big_set:Set[int] = set()
        for node in consider_set:
            # print("considering",node)
            if not self.graph.vp.f[node] or node in (0,1):
                continue
            neighset = set(self.view.iter_all_neighbors(node))
            # for neigh in neighset:   # Remove dead edges
            #     if neigh in (0,1):
            #         continue
            #     if ((self.view.edge(node,self.terminals[0]) and self.view.edge(neigh,self.terminals[0])) or 
            #             (self.view.edge(node,self.terminals[1]) and self.view.edge(neigh,self.terminals[1]))): # Remove useless edges
            #         if self.view.edge(node,neigh):
            #             # print(f"Removed edge from {node} to {neigh}")
            #             self.view.remove_edge(self.view.edge(node,neigh))
            #             big_set.add(neigh)

            if is_fully_connected(self.view,neighset):  # Dead nodes
                if iterate:
                    big_set.update(neighset)
                self.make_move(node,force_color="b")
                # print(f"{node} is dead")
                continue
            one_neighbors_neighbors = self.view.iter_all_neighbors(next(iter(neighset)))
            made_move = False
            for neighbor in one_neighbors_neighbors:
                if neighbor in (0,1) or neighbor==node:
                    continue
                without_me = set(self.view.get_all_neighbors(neighbor))-{node}
                without_him = neighset-{neighbor}
                if without_me == without_him:  # Maker captures
                    # print(f"maker captured {node}, {neighbor}")
                    self.response_set_maker[node] = neighbor
                    self.response_set_maker[neighbor] = node
                    self.make_move(neighbor,force_color="b")
                    change_set = self.make_move(node,force_color="m")
                    if iterate:
                        # big_set.update(without_him) # Should this be included?
                        big_set.update(without_me)
                        big_set.update(change_set)
                    made_move=True
                    break
            if made_move:
                continue

            for neighbor in neighset:
                if neighbor in (0,1):
                    continue
                without_me = set(self.view.get_all_neighbors(neighbor))-{node}
                without_him = neighset-{neighbor}
                if is_fully_connected(self.view,without_me) and is_fully_connected(self.view,without_him): # Breaker captures
                    self.response_set_breaker[node] = neighbor
                    self.response_set_breaker[neighbor] = node
                    # print(f"breaker captured {node}, {neighbor}")
                    if iterate:
                        big_set.update(without_me)
                        big_set.update(without_him)
                    self.make_move(node,force_color="b")
                    self.make_move(neighbor,force_color="b")
                    break
        big_set = big_set
        if iterate and len(big_set)>0:
            self.dead_and_captured(big_set,iterate=True)


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

    def copy(self,withboard=True):
        if not hasattr(self,"board") or self.board is None or not withboard:
            g = Node_switching_game()
            g.graph = Graph(self.graph)
            g.graph.vp.f = g.graph.new_vertex_property("bool")
            g.graph.vp.f.a = self.graph.vp.f.a.copy()
            g.view = GraphView(g.graph,g.graph.vp.f)
            g.terminals = [g.view.vertex(0),g.view.vertex(1)]
        else:
            new_board = self.board.copy()
            g = Node_switching_game()
            g.graph = Graph(self.graph)
            g.view = GraphView(g.graph,g.graph.vp.f)
            new_board.game = g
            new_board.vertex_to_board_index = {g.graph.vertex(int(key)):value for key,value in self.board.vertex_to_board_index.items()}
            new_board.board_index_to_vertex = {key:g.graph.vertex(int(value)) for key,value in self.board.board_index_to_vertex.items()}
            g.terminals = [g.graph.vertex(0),g.graph.vertex(1)]
            g.name = self.name
            g.board = new_board
            if self.board_callback is not None:
                g.board_callback = new_board.graph_callback
            g.response_set_breaker = self.response_set_breaker.copy()
            g.response_set_maker = self.response_set_maker.copy()
            g.callback_everything = self.callback_everything
            g.terminals = [g.view.vertex(0),g.view.vertex(1)]

        return g

    def set_to_graph(self,graph:Graph): # Do not use graph view. Use the graph itself.
        self.graph = graph
        self.terminals = [self.graph.vertex(0),self.graph.vertex(1)]
        self.view = GraphView(self.graph,vfilt=self.graph.vp.f)
        self.board = None
        self.name = "Shannon_node_switching_game"

    @staticmethod
    def from_graph(graph:Graph): # Do not use graph view. Use the graph itself.
        g = Node_switching_game()
        g.set_to_graph(graph)
        return g

    def prune_irrelevant_subgraphs(self) -> bool:
        """Prune all subgraphs that are not connected to a terminal node.

        Subgraphs that are only reachable from a terminal node by going through the other terminal node
        will also be pruned. As a side effect this function will find out if the position is won for breaker.

        Returns:
            If the position is won for breaker
        """
        self.view.vp.f[self.terminals[1]] = False
        found_vertices1 = set(x[1] for x in dfs_iterator(self.view,source=self.terminals[0],array=True))
        self.view.vp.f[self.terminals[1]] = True
        self.view.vp.f[self.terminals[0]] = False
        found_vertices2 = set(x[1] for x in dfs_iterator(self.view,source=self.terminals[1],array=True))
        self.view.vp.f[self.terminals[0]] = True
        valid_vertices = found_vertices1.intersection(found_vertices2)
        valid_vertices.add(0)
        valid_vertices.add(1)

        leftovers = set(self.view.get_vertices())-set(valid_vertices)
        for vi in leftovers:
            self.make_move(vi,force_color="b")

        return self.view.num_vertices==2

    def compute_node_voltages_iterate(self,iterations:int,voltage=100):
        """Compute approximate node voltages by treating the graph as an electrical circuit and using an iterative algorithm

        Some voltage is applied at terminal node 1 and terminal node 0 is the sink.
        Each edge has 1 Ohm resistance. Using Kirchhoff's current law, we find the node voltages
        by iteratively setting each nodes voltage to the mean of it's neighbors.
        The complexity of this is O(k*N*m) for k iterations, N vertices and an average
        neighborhood size of m.

        Args:
            voltage: How many volts to be applied at terminal node 1
        Returns:
            A vertex property map for the vertex voltages
            The current at terminal node 0 in Ampere, can be used as evaluation function.
        """
        vprop = self.view.new_vertex_property("double")
        vprop.a = voltage/2
        vprop[self.terminals[0]] = 0
        vprop[self.terminals[1]] = voltage
        for _ in range(iterations):
            for v in self.view.vertices():
                if v in self.terminals:
                    continue
                vprop[v] = sum(vprop[x] for x in v.all_neighbors())/v.out_degree()
        value = sum(x[1] for x in self.view.iter_all_neighbors(0,[vprop]))
        return vprop, value

    def compute_node_voltages_exact(self, voltage=100) -> Tuple[VertexPropertyMap,float]:
        """Compute exact node voltages by treating the graph as an electrical circuit

        Some voltage is applied at terminal node 1 and terminal node 0 is the sink.
        Each edge has 1 Ohm resistance. Using Kirchhoff's current law, we solve for
        the node voltages using a linear equation system. The complexity of this is
        O(N^3) for a graph with N vertices.

        Args:
            voltage: How many volts to be applied at terminal node 1
        Returns:
            A vertex property map for the vertex voltages
            The current at terminal node 0 in Ampere, can be used as evaluation function.
        """
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
        b[i2] = voltage
        voltages = scipy.linalg.solve(adj,b,overwrite_a=True,overwrite_b=True)
        v_prop = self.view.new_vertex_property("double")
        v_prop.fa = voltages
        value = sum(x[1] for x in self.view.iter_all_neighbors(0,[v_prop]))
        return v_prop,value

    def compute_node_currents(self,voltage_prop,check_validity=True) -> VertexPropertyMap:
        """Compute the currents flowing through each node in the electical circuit representation in the graph

        According to Kirchoffs laws, we can compute the current at each node by summing the voltage diffences to
        all neighbors with higher voltage. This has a runtime of O(N*m) for N vertices and average neighborhood size m.

        Args:
            voltage_prop: Property map containing vertex voltages.
            check_validity: If true, check if outgoing and incoming current for each node are the
                            same and raise error otherwise. This can happen if the node voltages
                            are not exact.
        """
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
                if check_validity:
                    drop = drop_low
                    if not np.isclose(drop_low,drop_high):
                        self.draw_me("error_graph.pdf",voltage_prop)
                    assert np.isclose(drop_low,drop_high)
                else:
                    drop = (drop_low+drop_high)/2
                d_prop[vertex] = drop
        return d_prop



    def draw_me(self,fname="node_switching.pdf",vprop1=None,vprop2=None,decimal_places=0,layout="grid"):
        """Draw the state of the graph and save it into a pdf file.

        Args:
            fname: Filename to save to
            vprop: Optional vertex property map of type int or double to display
        """
        if vprop1 is None:
            vprop = self.view.vertex_index
        elif vprop2 is None:
            vprop = self.view.new_vertex_property("string")
            for vertex in self.view.vertices():
                vprop[vertex] = str(vprop1[vertex])[:decimal_places+2]
        else:
            vprop = self.view.new_vertex_property("string")
            lower1 = vprop1.a.max()<1 and vprop2.a.max()<1
            for v in self.view.iter_vertices():
                if decimal_places == 0:
                    p1 = int(round(vprop1[v],0))
                    p2 = int(round(vprop2[v],0))
                else:
                    p1 = round(vprop1[v],decimal_places)
                    p2 = round(vprop2[v],decimal_places)
                    if lower1:
                        p1 = str(p1)[2:]
                        p2 = str(p2)[2:]
                    
                vprop[v] = str(p1)+"/"+str(p2)
            vprop[0] = ""
            vprop[1] = ""
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
        if layout=="grid" and hasattr(self,"board") and self.board is not None:
            layout = self.board.get_grid_layout()
        else:
            layout = sfdp_layout(self.view)

        graph_draw(self.view, pos=layout, vprops=vprops, vertex_text=vprop, output=fname)
