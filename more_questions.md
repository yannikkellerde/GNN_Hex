# Status
My current env is too slow to run any MCTS on interesting problems. Although this is a research project and not production code, I don't think I can get around spending a lot of time optimizing.

## Things are slow!
Experiment:
+ 128 MCTS are run for 200 Iterations each on 8x8 hex graph
+ Batched processing 128 game states at a time on GNN running on GPU
+ 50 hidden channels, 20 layers

hash 12.76779791075387  
process results 2.0024678500485606  
make move 25.61043147117016  
convert graph 19.889182167840772  
nn predictions 2.938752555011888  

+ hash time is to give unique hash to each graph in MCTS. Can be optimized fairly easily, I think. If not, it can be completely removed with different design of MCTS.
+ process results: This is to update the values in mcts. Although my env is not involved here, this takes almost as long as nn prediction.
+ Time to make moves in env is too slow.
  - Can be cut to ~ 1/4 if no dead and captured analysis, but that increases game lengths and average graph sizes by a lot.
	- Hard to optimize staying with python + Graph-Tool.
	- Dead and captured anaylsis could be sped up by un-generalizing. Only look for cases that are actually possible in hex-graphs and not for general shannon-node-switching games. Maybe even do template matching instead, although that's pretty sad for my graph representation.
	- Maybe my graph filtering approach is also at fault here. I need to really think about the underlying Boost Graphing Library structure and how to use it optimally. Maybe removing vertices will be faster than filtering out.
+ Convert graph: Converting from graph-tool representation of my game to representation that my pytorch-geometric model can process takes far too long. The reason for this is that because I currently use graph-filtering, the vertex indices in graph-tool are not continuous and I have to reindex all edge indices. I could and probably should delete vertices instead of filtering out. However this makes some other things more difficult such as tracking the correspondence between vertices and board states for visualization.
+ NN prediction: In total these are 200 GNN predictions, with batches of 128 graphs each. Comparably very fast. We should aim to make this the bottleneck.

https://github.com/bhansconnect/fast-alphazero-general show how alpha-zero can be done with multiprocessing in python. However, it will be more tricky with GNNs, based on the way batching works in pytorch-geometric. Also, even with 8 or 16 cpu threads, running the env will still be the major bottleneck.

+ We can reduce the amount of required make\_moves in MCTS significantly by caching the graphs at each node. However this requires us to copy a lot of graphs, which is also expensive.
+ Graph-tool includes a information on how to extend it's c++ backend https://graph-tool.skewed.de/static/doc/demos/cppextensions/cppextensions.html. It may be possible to write the make\_move and remove\_dead\_and\_captured function in c++ while keeping the rest of the code in python.

There is a lot to optimize before we can run any meaningful experiments. Also, I think there is no way around cpu multiprocessing in the end. There are a few things I can think of optimizing in the current python implementation, but it might also make sense to implement some parts in C++.


# Mostly solved questions
+ Alpha-zero starts with a temperature 1 in the beginning of the game and then drops to zero after n moves. This makes sense, because exploration is more valuable in the beginning. However, from the paper and some reference implementations it seems like the temperature is also variied for the training targets. E.g. for the first n moves, the network is supposed to predict a distribution and for later moves is is supposed to predict 100% for the best move. This sounds like it would make training unstable. Why not use some fixed temperature for computing the training targets and only switch up the temperature for self-play?
+ The paper is a little unclear about which transitions should be used for training (And when should old transitions be thrown out). Just coninuously collect transitions and throw oldest ones out when some capacity is reached? Or aim to only train on data from newest agent?
+ MCTS:
	- There are two ways to implement MCTS:
		1. My first attempt: Build up tree of nodes that each cache the graphs and store visits, priors, q etc.
			* pro: Using caching, the same make_move never has to be executed twice in the MCTS
			* con: We need to copy the graph to restore the game state each time when we want expand the game state. Maybe faster alternative would be a reversible make_move function.
		2. Adapted from reference implementations: store visits, priors, q etc. in dictionary based on unique hash of position.
			* pro: No graph copying required. Fewer lines and less convoluted code. No need to store potentially many graphs in memory.
			* con: Need to remake the move each time when traversing the graph. Need to create unique hash of each position. Isomorphism detection via hashing sounds great, but does not work easy because otherwise order of actions (vertex indices) is not well defined. Thus, we hash avoiding isomorphisms.
	- Currently the first attempt is only faster for very deep MCTS trees. Otherwise second is faster.
	- What is the range of v from the neural network? In the paper it says that it is the probability of the current player winner (0-1), but at some other places and in reference impl. it seems like it is -1 for lost games.
	- Node that just got expanded have 0 or 1 visits? I think 1 makes more sense, but reference impl uses 0

+ GNN:
	- My input features to the GNN: Currently, 3D, one dim for *is neighbor of termial node 1*, one dim for *is neighbor of terminal node 2* and one dim for *Is it makers turn*. Should I add more? E.g. degree? Is it sensible to add is\_makers\_turn as an additional feature dimension to all nodes or is there a smarter way?
