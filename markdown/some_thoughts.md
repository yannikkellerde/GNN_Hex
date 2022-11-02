# What I have done
## Before last meeting
+ Implement Graph Representation for Qango/Tic-Tac-Toe/Go-Moku (winnpattern game)
+ Solve Qango using PN-DAG and Threat-space-search [https://python.yannikkeller.de/solver/analyze/qango6x6?ruleset=tournament](https://python.yannikkeller.de/solver/analyze/qango6x6?ruleset=tournament)
+ Train a Graph Neural Network to simulate Threat-space-search on Qango 6x6 (96% Accuracy, requires graph-level attributes)
	- Performance drops significantly when trying to transfer to Tic-Tac-Toe or other Qango sizes.

## Since last meeting
+ Implement Graph Representation for Hex (Shannon-node-switching game)
+ Implement some basic captured/dead cell analysis, working directly on the graph representation.
+ Implement Shannons electrical circuit algorithm usidng the exact linear programming approach as well as iterative approach. (Simple evaluation function for hex)
+ Train Graph Neural Networks with policy and value head to reproduce electrical circuit algorithm.
	- Using GraphSAGE + GraphNorm
	- Works really well, 0.26 MSE on node voltages with 100 Volt applied, with 13 message passing steps, 32 hidden size, on 11x11 board.
	- Predicts position evaluation function (total current flow) almost perfectly (eg. 0.00001 MSE)
	- Outperforms iterative approach in terms of message passing steps vs MSE.
	- Native transfer to smaller board, performance slowly drops when increasing board size.
+ Try out some GNN architectures (see tensorboard). Main results:
	- ResNet style architectures are superior (e.g. GraphSAGE).
	- GraphNorm improves convergence speed and stability a lot, but also makes network passes take twice as much time. (Not sure why it is so much)
	- A lot helps a lot, e.g. more parameters generally improve convergence speed and total performance.
	- Training using MSE on node voltages turned out working better than using softmax and cross-entropy in this case. This may however have to do with my lack of temperature tuning (For node voltages, the softmax actually acted close to a "max" operator).


# Some thoughts about how to continue
### Initialization/Transfer
Maybe initialize the network based on supervised learning on the electrical circuit before going into MCTS?

### C(++)
I do think using pytorch\_geometric is pretty essential to be able to build on work from others and not start from scratch.
When running MCTS it will become clear if network passes or the generation of next positions and converting them into suitable format (playing out moves) will be more of a bottleneck.
If generating the positions is a significant bottleneck, I have two options:
1. I could [write extensions for graph-tool in C++](https://graph-tool.skewed.de/static/doc/demos/cppextensions/cppextensions.html) to speed up some of my subroutines such as pruning captured and dead cells or converting to pytorch\_geometric format.
2. I could ditch graph-tool and my current implementation of hex and rewrite the whole game logic in C(++) and only use python for MCTS logic and pytorch. [This](https://github.com/richemslie/galvanise_zero) project does seem to do something similar.

I did however find a strong hex ai that is written only in python [https://github.com/harbecke/HexHex](https://github.com/harbecke/HexHex). This one does not use MCTS because it was "prohibitively expensive". Their alterantive method is defenitely worth looking into though, because their ai is acutally really strong.

### MCTS and Batching
For the Graph Net policy and value approximation to be efficient, we need to batch many graphs together. Thus, we either have to play multiple games at once or do the MCTS in an inexact way (e.g. doing new expansions before others where evaluated). There are some approaches in the AlphaGo/AlphaZero papers (APV-MCTS, virtual loss), but there are decisions to be made.

It will show itself if CPU computation of positions and tree traversal or GPU policy and value approximation is more of a bottleneck. If the first is the case, it will be usefull to implement MCTS in a multiprocessing fashion.

### Where do we stand in the world of hex?
There have been multiple approaches at transfering the AlphaZero method to hex. The strongest hex ai ever build is likely [galvanise\_zero](https://github.com/richemslie/galvanise_zero). According to most people in the hex community, it was already at superhuman level. The GNN approach however stands out as something that I haven't found any examples of being tried on hex.

### Curiculum?
Hex 6x6 is a reachable position from any larger Hex board. Knowledge of how to play 6x6 hex is thus verly likely to be usefull for playing on larger boards.
Maybe it makes sense to train the hex agent in a curriculum with a growing network size: start by learning 6x6 hex with a small network with not a lot of 
passing steps. After we master this, grow to 7x7, transfer parameters but add a passing step and network width. Iterate to desired hex size (e.g. 13x13).

### Pyramid style GNNs?
Dense neural networks are known to perform better with a pyramid style architecture. E.g. start with large hidden size and shrink each layer. Does this also make sense for GNNs?
