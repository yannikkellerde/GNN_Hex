# Fast shannon node-switching game in C++

Key Files are:
+ `graph.cpp`: My own implementation of a graph structure. Layout makes it quick to add / remove nodes and to convert to a format that can be processed by the torch\_geometric neural networks.
+ `shannon_node_switching_game.cpp`: Handles the graph representation using my own graph structure from `graph.cpp`. E.g. making moves, removing dead and captured nodes etc.
+ `hex_board_game.cpp`: Small and plain hex grid representation.
+ `util.cpp`: Utilities for creating swapmaps and batching of graph data for GNN processing.

Subfolders include:
+ `tests`: Intensive tests for correctness and speed of the graph structure itself and the hex graph game.
+ `main`: An alternative entrypoint for tests on the C++ shannon node-switching game without HexAra
