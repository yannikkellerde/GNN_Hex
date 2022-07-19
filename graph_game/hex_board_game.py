from graph_game.abstract_board_game import Abstract_board_game
from typing import List
from blessings import Terminal
import math

class Hex_board(Abstract_board_game):
    game:"Node_switching_game"
    position:List[str]

    def __init__(self):
        pass

    def make_move(self, move:int):
        """Make a move on the board representation and update the graph representation.
        
        Args:
            move: The square the move is to be made on."""
        self.position[move] = self.onturn
        self.onturn = "r" if self.onturn == "b" else "r"
        self.game.graph_from_board()      

    def pos_from_graph(self):
        pass

    def draw_me(self,pos=None):
        out_str = ""
        t = Terminal()
        if pos is None:
            pos=self.position
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
    pos = ["f"]*bgame.squares
    for _ in range(8*8):
        pos[random.randint(0,len(pos)-1)] = random.choice(["b","r"])
    print(bgame.draw_me(pos))

