# What I have done
## Before last meeting
+ Implement Graph Representation for Qango/Tic-Tac-Toe/Go-Moku (winnpattern game)
+ Solve Qango using PN-DAG and Threat-space-search [https://python.yannikkeller.de/solver/analyze/qango6x6?ruleset=tournament](https://python.yannikkeller.de/solver/analyze/qango6x6?ruleset=tournament)
+ Train a Graph Neural Network to simulate Threat-space-search (98% Accuracy, requires graph-level attributes)

## Since last meeting
+ Implement Graph Representation for Hex (Shannon-node-switching game)
+ Implement some basic captured/dead cell analysis, working directly on the graph representation.
+ Implement Shannons electrical circuit algorithm. (exact linear programming approach as well as iterative approach)
+ Train Graph Neural Network to reproduce electrical circuit algorithm.
	- Using GraphSAGE + GraphNorm
	- Works really well, 0.26 MSE with 100 Volt applied, with 13 message passing steps on 11x11 board.
	- Outperforms iterative approach in terms of message passing steps vs MSE.
	- Native transfer to smaller board, performance slowly drops when increasing board size.
	- Similar to other norms, GraphNorm uses batch statistics to normalize layers.
		* This makes some trouble during testing of the Network. E.g. Evaluating on smaller batch sizes (e.g. on single graph) than during training hurts performance.
		* Also, evaluating on batch of graphs from hex boards of smaller size than during training destroys batch statistics and kills performance.
			Thus currently, we only get performance on smaller board if we embed it in a batch of graphs from training distribution.
		* Might just be an implementation issue and can be fixed by smarter caching of batch statistics. E.g. I'll have to modify/inherit some more torch_geometric code.
		* However, I had this problem before on other projects using BatchNorm. Wondering what are the smart ways to deal with this...


# Some thoughts about how to continue
### Initialization/Transfer
Maybe initialize the network based on supervised learning on the electrical circuit before going into MCTS?
### Curiculum?
Hex 6x6 is a reachable position from any larger Hex board. Knowledge of how to play 6x6 hex is thus verly likely to be usefull for playing on larger boards.
Maybe it makes sense to train the hex agent in a curriculum with a growing network size: start by learning 6x6 hex with a small network with not a lot of 
passing steps. After we master this, grow to 7x7, transfer parameters but add a passing step and network width. Iterate to desired hex size (e.g. 13x13).
### Speed and Programming Language
I am not scared of C. However, I think it makes sense to use python not only because I am more comfortable with it, but also to use pytorch-geometric and
other modules to build upon in python.

However, I am aware that python can be slow at times and during MCTS this can be a heavy limiting factor. The graph library that I use, graph-tool, has it's core data
structures and algorithms implemented in C++. It is possible to [write extensions for it in C++](https://graph-tool.skewed.de/static/doc/demos/cppextensions/cppextensions.html).
If it becomes nescessary, I could rewrite some functionalitly of my hex graph as a C++ extension (such as removing dead and captured cells) to decrease runtime.
### MCTS and Batching
For the Graph Net policy and value approximation to be efficient, we need to batch many graphs together. Thus, we either have to play multiple games at once or do the MCTS in an inexact way (e.g. doing new expansions before others where evaluated). There are some approaches in the AlphaGo/AlphaZero papers, but there are decisions to be made.

It will show itself if CPU computation of positions and tree traversal or GPU policy and value approximation is more of a bottleneck. If the first is the case, it may be usefull to implement MCTS in a multiprocessing fashion.
