from graph_game.abstract_board_game import Abstract_board_game
import numpy as np
from graph_tool.all import Graph,Vertex,VertexPropertyMap,GraphView
from typing import List,Dict
from blessings import Terminal
from graph_game.utils import fully_connect_lists,take_step,greedy_search
import math

class Hex_board(Abstract_board_game):
    game:"Node_switching_game"
    position:List[str]
    node_map:Dict[int,int]

    def __init__(self):
        pass

    def make_move(self, move:int):
        """Make a move on the board representation and update the graph representation.
        
        Args:
            move: The square the move is to be made on."""
        self.position[move] = self.onturn
        self.onturn = "r" if self.onturn == "b" else "r"
        self.graph_from_board(True)      

    def grid_to_double_triangle(self,move:int):
        """Transform a move with grid numbering to a move with double triangle numbering"""
        sq_squares = int(math.sqrt(self.squares))
        row = int(move//sq_squares)
        col = int(move%sq_squares)
        decend = row+col+1
        before_num = sum(i for i in range(min(sq_squares,decend)))+sum(i for i in range(sq_squares,2*sq_squares-decend,-1))
        in_row_num = col-(0 if decend<sq_squares else decend-sq_squares)
        return before_num+in_row_num

    def transform_position_to_double_triangle(self,pos:List[str]):
        new_pos = pos.copy()
        for i in range(len(pos)):
            new_pos[self.grid_to_double_triangle(i)] = pos[i]
        return new_pos

    def transform_position_from_double_triangle(self,pos:List[str]):
        new_pos = pos.copy()
        for i in range(len(pos)):
            new_pos[i] = pos[self.grid_to_double_triangle(i)]
        return new_pos
        
    @staticmethod
    def evaluate_graph_similarity(graph1,graph2,node_map1,node_map2):
        cost = 0
        inv_node_map1 = {value:key for key,value in node_map1.items()}
        inv_node_map2 = {value:key for key,value in node_map2.items()}
        for e in graph1.edges():
            s = int(e.source())
            t = int(e.target())
            s_mapped = s if s<2 else inv_node_map2[node_map1[s]]
            t_mapped = t if t<2 else inv_node_map2[node_map1[t]]
            if graph2.edge(s_mapped,t_mapped) is None:
                cost += 1
        
        for e in graph2.edges():
            s = int(e.source())
            t = int(e.target())
            s_mapped = s if s<2 else inv_node_map1[node_map2[s]]
            t_mapped = t if t<2 else inv_node_map1[node_map2[t]]
            if graph1.edge(s_mapped,t_mapped) is None:
                cost += 1
        return cost

    def pos_from_graph(self,redgraph:bool):
        str_map = {0:"U",1:"f",2:"r",3:"b"}
        step_take_obj = take_step([2,3])

        def evaluate_assignment(assignment):
            new_pos = known_pos.copy()
            new_pos[new_pos==0] = assignment
            graph,node_map,_ = self.board_to_graph([str_map[x] for x in new_pos],redgraph)
            cost = Hex_board.evaluate_graph_similarity(graph,self.game.graph,node_map,self.node_map)
            return cost

        known_pos = np.zeros(self.squares) #0:unknown,1:empty,2:red,3:blue
        for v in self.game.view.vertices():
            if v not in self.game.terminals:
                known_pos[self.node_map[int(v)]] = 1
        initial_assignment = np.ones(self.squares-self.game.view.num_vertices()+2)*(3 if redgraph else 2)
        res,_fun_val = greedy_search(evaluate_assignment,initial_assignment,step_take_obj)
        known_pos[known_pos==0] = res
        self.position = [str_map[x] for x in known_pos]

    @staticmethod
    def board_to_graph(position:List[str],redgraph:bool):
        squares = len(position)
        node_map = {}
        sq_squares = int(math.sqrt(squares))
        graph = Graph(directed=False)
        terminals = [graph.add_vertex(),graph.add_vertex()]
        references = {i:set() for i in range(squares)}
        for i in range(squares):
            #print({key:[int(x) for x in value] for key,value in references.items()})
            if (position[i]=="b" and redgraph) or (not redgraph and position[i]=="r"):
                continue
            elif (position[i]=="r" and redgraph) or (not redgraph and position[i]=="b"):
                connecto = True
            else:
                v = graph.add_vertex()
                references[i] = [v]
                node_map[int(v)]=i
                connecto = False

            if (i<sq_squares and redgraph) or (not redgraph and i%sq_squares==0):
                if connecto:
                    references[i].add(terminals[0])
                else:
                    graph.add_edge(v,terminals[0])
            if (i//sq_squares==sq_squares-1 and redgraph) or (not redgraph and i%sq_squares==sq_squares-1):
                if connecto:
                    references[i].add(terminals[1])
                else:
                    graph.add_edge(v,terminals[1])
            if i%sq_squares>0:
                if connecto:
                    references[i].update(references[i-1])
                else:
                    fully_connect_lists(graph,[v],references[i-1])
            if i>=sq_squares:
                if connecto:
                    references[i].update(references[i-sq_squares])
                else:
                    fully_connect_lists(graph,[v],references[i-sq_squares])
                if i%sq_squares!=sq_squares-1:
                    if connecto:
                        references[i].update(references[i-sq_squares+1])
                    else:
                        fully_connect_lists(graph,[v],references[i-sq_squares+1])
            if connecto:
                j = i-1
                while j%sq_squares>0 and (position[j]=="r" and redgraph) or (position[j]=="b" and not redgraph):
                    references[j] = references[i]
                    j-=1
                    
                fully_connect_lists(graph,references[i],references[i])
        return graph,node_map,terminals


    def graph_from_board(self, redgraph:bool): # To test ...
        self.game.graph,self.node_map,self.game.terminals = Hex_board.board_to_graph(self.position,redgraph)
        filt_prop = self.game.graph.new_vertex_property("bool")
        self.game.graph.vp.f = filt_prop # For filtering in the GraphView
        self.game.graph.vp.f.a = np.ones(self.game.graph.num_vertices()).astype(bool)
        self.game.view = GraphView(self.game.graph,self.game.graph.vp.f)


    def draw_me(self,pos=None):
        out_str = ""
        t = Terminal()
        if pos is None:
            pos=self.transform_position_to_double_triangle(self.position)
        sq_squares = int(math.sqrt(self.squares))
        before_spacing = sq_squares
        out_str+=" "*before_spacing+" "+t.magenta("_")+"\n"
        row_width=1
        row_index=0
        before_center=True
        out_str+=" "*before_spacing
        out_str+=t.blue("/")
        for p in pos:
            if p=="b":
                out_str+=t.blue("⬢")
            elif p=="r":
                out_str+=t.red("⬢")
            elif p=="f":
                out_str+=t.white("⬢")
            
            row_index+=1
            if row_index==row_width:
                if row_width==sq_squares:
                    before_center=False
                row_index = 0
                if before_center:
                    row_width+=1
                    before_spacing-=1
                    out_str+=t.red("\\")+"\n"
                    out_str+=before_spacing*" "
                    if row_width==sq_squares:
                        out_str+=t.magenta("|")
                    else:
                        out_str+=t.blue("/")
                else:
                    before_spacing+=1
                    if row_width==sq_squares:
                        out_str+=t.magenta("|")
                    else:
                        out_str+=t.blue("/")
                    row_width-=1
                    out_str+="\n"+before_spacing*" "
                    if row_width==0:
                        out_str+=t.magenta("‾")
                    else:
                        out_str+=t.red("\\")
            else:
                out_str+=" "
        return out_str

if __name__=="__main__":
    import random
    bgame = Hex_board()
    bgame.squares=11*11
    pos = list("fffffffffbf"
               "fffffffffbf"
               "fffffffffbf"
               "fffffffffbf"
               "rrrrrrrrrbr"
               "fffffffffbf"
               "fffffffffbf"
               "fffffffffbf"
               "fffffffffbf"
               "fffffffffbf"
               "fffffffffbf"
           )
    bgame.position=pos

    print(bgame.draw_me())
    while 1:
        m = int(input())
        print(bgame.grid_to_double_triangle(m))

