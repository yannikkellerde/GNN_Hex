# Graph games

This folder implements the python games for this project, with graph and board representations. The functionalities of the files in this folder are designed to work independently from the rest of the project.

Key files implementing things talked about in the thesis are:
+ `shannon_node_switching_game.py`: This implements the behavior of the graph environment used in the thesis including the algorithm for dead and captured nodes removal
+ `hex_board_game.py`: This implements the conversion from Hex-grid representation to graph representation.

Other useful files are:
+ `hex_gui.py`: This implements an interactive Hex gui implemented in matplotlib that can be used to play against NN models or investigate the behavior of the graph structure.
+ `winpattern_board.py`, `winpattern_game.py` are files from before this thesis start that implement a graph representation for Tic-Tac-Toe and Qango.

