""" A class representing a Hex board in grid representation. It can optionally be connected with a graph representation via the 'game' attribute

Key functions include:
    graph_from_board: Converts a Hex board representation in a graph representation
    matplotlib_me: Plot the board state with optional square attributes in matplotlib.
    draw_me: Draw the board in the terminal.
"""

from graph_game.abstract_board_game import Abstract_board_game

import random
import numpy as np
from graph_tool.all import Graph,Vertex,VertexPropertyMap,GraphView
from typing import List,Dict,Union
from blessings import Terminal
from graph_game.utils import fully_connect_lists,take_step,greedy_search
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import RegularPolygon

class Hex_board(Abstract_board_game):
    game:"Node_switching_game"
    position:List[str]
    board_index_to_vertex:Dict[int,Vertex]
    vertex_to_board_index:Dict[Vertex,int]
    vertex_index_to_board_index:Dict[int,int]
    board_index_to_vertex_index:Dict[int,int]
    redgraph:bool
    size:int

    def __init__(self,onturn="r",redgraph=True):
        self.onturn = onturn
        self.redgraph = redgraph

    def transpose_move(self,move):
        x = move%self.size
        y = move//self.size
        return x*self.size+y

    def copy(self):
        new_board = Hex_board(onturn=self.onturn,redgraph=self.redgraph)
        new_board.position = self.position.copy()
        new_board.squares = self.size**2
        new_board.size = self.size
        new_board.vertex_index_to_board_index = self.vertex_index_to_board_index.copy()
        new_board.board_index_to_vertex_index = self.board_index_to_vertex_index.copy()
        return new_board

    def to_gao_input_planes(self,*args,**kwargs):
        red_plane_c = torch.tensor([1 if p=="r" else 0 for p in self.position],dtype=torch.float).reshape((self.size,self.size))
        blue_plane_c = torch.tensor([1 if p=="b" else 0 for p in self.position],dtype=torch.float).reshape((self.size,self.size))
        empty_plane_c = torch.tensor([1 if p=="f" else 0 for p in self.position],dtype=torch.float).reshape((self.size,self.size))

        red_plane = torch.zeros((self.size+2,self.size+2))
        blue_plane = torch.zeros((self.size+2,self.size+2))
        empty_plane = torch.zeros((self.size+2,self.size+2))

        red_plane[0,:] = 1
        red_plane[-1,:] = 1
        blue_plane[:,0] = 1
        blue_plane[:,-1] = 1
        red_plane[1:-1,1:-1] = red_plane_c 
        blue_plane[1:-1,1:-1] = blue_plane_c 
        empty_plane[1:-1,1:-1] = empty_plane_c

        if self.game.view.gp["m"]:
            onturn_plane = torch.ones_like(red_plane)
        else:
            onturn_plane = torch.zeros_like(red_plane)

        return torch.stack((red_plane,blue_plane,onturn_plane,empty_plane))
        

    def to_input_planes(self,hex_size=None,zero_fill=False,flip=False):
        if hex_size is None:
            hex_size = self.size
        assert hex_size>=self.size
        red_plane = torch.tensor([1 if p=="r" else 0 for p in self.position],dtype=torch.float).reshape((self.size,self.size))
        blue_plane = torch.tensor([1 if p=="b" else 0 for p in self.position],dtype=torch.float).reshape((self.size,self.size))
        if hex_size > self.size:
            new_red_plane = torch.zeros((hex_size,hex_size),dtype=torch.float)
            new_blue_plane = torch.zeros((hex_size,hex_size),dtype=torch.float)
            new_red_plane[:self.size,:self.size] = red_plane
            new_blue_plane[:self.size,:self.size] = blue_plane
            if not zero_fill:
                new_red_plane[:self.size,self.size:] = 1
                new_blue_plane[self.size:,:self.size] = 1
            red_plane = new_red_plane
            blue_plane = new_blue_plane
        if self.game.view.gp["m"]:
            onturn_plane = torch.ones_like(red_plane)
        else:
            onturn_plane = torch.zeros_like(red_plane)
        if flip:
            return torch.stack((red_plane,blue_plane,onturn_plane,torch.flip(red_plane,dims=[0]),torch.flip(blue_plane,dims=[0])))
        else:
            return torch.stack((red_plane,blue_plane,onturn_plane))

    def sample_legal_move(self):
        free_squares = []
        for i in range(len(self.position)):
            if self.position[i] == "f":
                free_squares.append(i)
        return random.choice(free_squares)

    def vertex_index_to_string_move(self,vi):
        board_index = self.vertex_index_to_board_index[vi]
        letters = "abcdefghikjlmnopqrstuvwxyz"
        return letters[board_index%self.size]+str(board_index//self.size+1)

    def fill_dead_and_captured(self):
        for key,value in self.game.response_set_maker.items():
            self.position[self.vertex_index_to_board_index[int(key)]] = "r"
            self.position[self.vertex_index_to_board_index[int(value)]] = "b"

        for key,value in self.game.response_set_breaker.items():
            self.position[self.vertex_index_to_board_index[int(key)]] = "b"
            self.position[self.vertex_index_to_board_index[int(value)]] = "b"

        for i in range(len(self.position)):
            if self.position[i]=="f" and not self.game.graph.vp.f[self.board_index_to_vertex[i]]:
                self.position[i] = "b"
        
    def get_all_unique_starting_moves(self):
        unique_moves = []
        for i in range(self.size):
            unique_moves.extend(list(range(i+i*self.size,self.size+i*self.size)))
        return unique_moves


    def get_actions(self):
        return [i for i in range(len(self.position)) if self.position[i]=="f"]

    def make_move(self, move:int, force_color=None, remove_dead_and_captured=False, only_legal=True):
        """Make a move on the board representation and update the graph representation.
        
        Args:
            move: The square the move is to be made on."""
        color = self.onturn if force_color is None else force_color
        if only_legal and self.position[move]!="f":
            return "illegal"
        self.position[move] = color
        if force_color is None:
            self.onturn = "r" if color == "b" else "b"
            to_force = None
        else:
            if (force_color=="r" and self.redgraph) or (force_color=="b" and not self.redgraph):
                to_force = "m"
            else:
                to_force = "b"
        if not self.game.graph.vp.f[self.board_index_to_vertex[move]] and only_legal:
            self.game.view.gp["m"] = not self.game.view.gp["m"]
            return "only board"
        self.game.make_move(self.board_index_to_vertex[move],force_color=to_force,remove_dead_and_captured=remove_dead_and_captured)
        return "legal"

    def graph_callback(self, vertex_move:int, makerturn:bool):
        color = "r" if makerturn==self.redgraph else "b"
        if self.position[self.vertex_to_board_index[vertex_move]]=="f":
            self.position[self.vertex_to_board_index[vertex_move]] = color

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
            s = e.source()
            t = e.target()
            s_mapped = s if int(s)<2 else inv_node_map2[node_map1[s]]
            t_mapped = t if int(t)<2 else inv_node_map2[node_map1[t]]
            if graph2.edge(s_mapped,t_mapped) is None:
                cost += 1
        
        for e in graph2.edges():
            s = e.source()
            t = e.target()
            s_mapped = s if int(s)<2 else inv_node_map1[node_map2[s]]
            t_mapped = t if int(t)<2 else inv_node_map1[node_map2[t]]
            if graph1.edge(s_mapped,t_mapped) is None:
                cost += 1
        return cost

    def graph_from_board(self, redgraph:bool, no_worthless_edges=True):
        self.redgraph=redgraph
        sq_squares = int(math.sqrt(self.squares))
        self.board_index_to_vertex = {}
        self.game.graph = Graph(directed=False)
        self.game.terminals = [self.game.graph.add_vertex(),self.game.graph.add_vertex()]
        for i in range(self.squares):
            v = self.game.graph.add_vertex()
            self.board_index_to_vertex[i] = v
            if (i<sq_squares and redgraph) or (not redgraph and i%sq_squares==0):
                self.game.graph.add_edge(v,self.game.terminals[0])
            if (i//sq_squares==sq_squares-1 and redgraph) or (not redgraph and i%sq_squares==sq_squares-1):
                self.game.graph.add_edge(v,self.game.terminals[1])
            if i%sq_squares>0 and ((not no_worthless_edges) or (sq_squares<=i<=self.squares-sq_squares)):
                self.game.graph.add_edge(v,self.board_index_to_vertex[i-1])
            if i>=sq_squares:
                self.game.graph.add_edge(v,self.board_index_to_vertex[i-sq_squares])
                if i%sq_squares!=sq_squares-1:
                    self.game.graph.add_edge(v,self.board_index_to_vertex[i-sq_squares+1])

        self.vertex_to_board_index = {value:key for key,value in self.board_index_to_vertex.items()}
        self.game.graph.gp["m"] = self.game.graph.new_graph_property("bool")
        self.game.graph.gp["m"] = True
        filt_prop = self.game.graph.new_vertex_property("bool")
        self.game.graph.vp.f = filt_prop # For filtering in the GraphView
        self.game.graph.vp.f.a = np.ones(self.game.graph.num_vertices()).astype(bool)
        self.game.view = GraphView(self.game.graph,self.game.graph.vp.f)

        self.vertex_index_to_board_index = {int(key):value for key,value in self.vertex_to_board_index.items()}
        self.board_index_to_vertex_index = {value:key for key,value in self.vertex_index_to_board_index.items()}

        for i in range(self.squares):
            if (self.position[i] == "r" and redgraph) or (self.position[i]=="b" and not redgraph):
                self.game.graph.gp["m"] = True
                self.game.make_move(self.board_index_to_vertex[i])
            elif self.position[i]!="f":
                self.game.graph.gp["m"] = False
                self.game.make_move(self.board_index_to_vertex[i])

    def matplotlib_me(self,vprop=None,color_based_on_vprop=False,fig=None):
        colors = [[("w" if y=="f" else y) for y in self.position[x:x+self.size]] for x in range(0,self.size*self.size,self.size)]
        labels = None
        if vprop is not None:
            labelist = [vprop[self.board_index_to_vertex[i]] for i in range(len(self.position))]
            labels = [["" if x == 0 else x for x in labelist[x:x+self.size]] for x in range(0,self.size**2,self.size)]
            if color_based_on_vprop:
                colors = [["g" if type(x)==str else ("b" if x<-0.1 else ("r" if x>0.1 else "w")) for x in y] for y in labels]
            labels = [[str(a)[:4] for a in x] for x in labels]

        return build_hex_grid(colors,labels,fig=fig,do_pause=False,fontsize=12)

    def number_to_notation(self,number):
        letters = "abcdefghijklmnopqrstuvwxyz"
        return letters[number//self.size]+str(number%self.size+1)

    def notation_to_number(self,notation):
        letters = "abcdefghijklmnopqrstuvwxyz"
        return letters.index(notation[0])*self.size+int(notation[1:])-1

    def sgf_from_move_history(self,move_history,starting_player,red=None,blue=None):
        if starting_player=="r":
            onturn = "W"
        else:
            onturn = "B"
        sgf = f"(;AP[RainbowHex]FF[4]GM[11]SZ[{self.size}]PL[{onturn}]"
        if red is not None:
            sgf += f"PW[{red}]"
        if blue is not None:
            sgf += f"PB[{blue}]"
        for move in move_history:
            sgf+=f"\n;{onturn}[{self.number_to_notation(move)}]"
            if onturn=="W":
                onturn = "B"
            else:
                onturn = "W"
        sgf+=")"
        return sgf

    def to_sgf(self):
        sgf = f"(;AP[RainbowHex]FF[4]GM[11]SZ[{self.size}]"
        sgf += "PL[W]" if self.game.view.gp["m"] else "PL[B]"
        sgf+=";AB"
        for i,p in enumerate(self.position):
            if p == "b":
                sgf+=f"[{self.number_to_notation(i)}]"
        sgf+=";AW"
        for i,p in enumerate(self.position):
            if p == "r":
                sgf+=f"[{self.number_to_notation(i)}]"
        sgf+=")"
        return sgf

    def draw_me(self,pos=None,green=False):
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
        c1 = t.red
        c2 = t.green if green else t.blue
        out_str+=" "*before_spacing
        out_str+=c2("/")
        for p in pos:
            if p=="b":
                out_str+=c2("⬢")
            elif p=="r":
                out_str+=c1("⬢")
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
                    out_str+=c1("\\")+"\n"
                    out_str+=before_spacing*" "
                    if row_width==sq_squares:
                        out_str+=t.magenta("|")
                    else:
                        out_str+=c2("/")
                else:
                    before_spacing+=1
                    if row_width==sq_squares:
                        out_str+=t.magenta("|")
                    else:
                        out_str+=c2("/")
                    row_width-=1
                    out_str+="\n"+before_spacing*" "
                    if row_width==0:
                        out_str+=t.magenta("‾")
                    else:
                        out_str+=c1("\\")
            else:
                out_str+=" "
        return out_str
    
    def get_grid_layout(self) -> VertexPropertyMap:
        vprop = self.game.view.new_vertex_property("vector<double>")
        scale = 400/self.size
        xstart = 0
        ystart = 0
        xend = xstart+1.5*(self.size-1)*scale
        yend = ystart+np.sqrt(3/4)*(self.size-1)*scale
        vprop[self.game.terminals[0]]=[xstart,yend/2]
        vprop[self.game.terminals[1]]=[xend,yend/2]
        for i in range(self.size):
            for j in range(self.size):
                coords = [xstart+(0.5*j+i)*scale,yend-(np.sqrt(3/4)*j)*scale]
                vprop[self.board_index_to_vertex[i*self.size+j]] = coords
        return vprop


def build_hex_grid(colors,labels=None,fig=None,border_swap=False,do_pause=True,fontsize=14):
    if labels is not None:
        labels = [[str(x)[:6] for x in y] for y in labels]
    if fig is not None:
        ax = fig.axes[0]
    else:
        fig, ax = plt.subplots(1,figsize=(16,16))
    size = len(colors)
    xstart = -(size//2)*1.5
    ystart = -(size/2*np.sqrt(3/4))+0.5
    xend = xstart+1.5*(size-1)
    yend = ystart+np.sqrt(3/4)*(size-1)
    ax.set_aspect('equal')
    tri = plt.Polygon([[0,0],[xstart-1.25,ystart-0.75],[xstart+0.5*(size-1)-0.5,yend+0.75]],color="b" if border_swap else "r",alpha=0.7)
    ax.add_patch(tri)
    tri = plt.Polygon([[0,0],[xend+1.25,yend+0.75],[xstart+0.5*(size-1)-0.5,yend+0.75]],color="r" if border_swap else "b",alpha=0.7)
    ax.add_patch(tri)
    tri = plt.Polygon([[0,0],[xend+1.25,yend+0.75],[xstart+1*(size-1)+0.5,ystart-0.75]],color="b" if border_swap else "r",alpha=0.7)
    ax.add_patch(tri)
    tri = plt.Polygon([[0,0],[xstart-1.25,ystart-0.75],[xstart+1*(size-1)+0.5,ystart-0.75]],color="r" if border_swap else "b",alpha=0.7)
    ax.add_patch(tri)
    for i,cylist in enumerate(colors):
        for j,color in enumerate(cylist):
            coords = [xstart+0.5*j+i,ystart+np.sqrt(3/4)*j]
            hexagon = RegularPolygon((coords[0], coords[1]), numVertices=6, radius=np.sqrt(1/3), alpha=1, edgecolor='k', facecolor=color,linewidth=2)
            if labels is not None:
                ax.text(coords[0]-0.4, coords[1]-0.05,labels[i][j],color="black" if (color=="white" or color=="w") else "white",fontsize=14)
            ax.add_patch(hexagon)
    plt.autoscale(enable=True)
    plt.axis("off")
    plt.tight_layout()
    if do_pause:
        plt.pause(0.001)
    return fig

if __name__=="__main__":
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

