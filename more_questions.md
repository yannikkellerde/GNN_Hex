# Status
<!-- **Bitter Truth: Alpha zero requires a very fast environment to collect enough training data through MCTS. With an efficient board representation, making a move comes down to a single bitwise xor. It is impossible to match that speed with a graph representation, no matter how well I optimize.** -->
<!-- + Question to Gopika: Do you need to manipulate graphs when training you GNNs? How do you represent them, what frameworks do you use? -->
<!-- + Even without dead and captured removal, we need to remove a vertex and connect all it's neighbors when the maker makes a move. -->
<!-- + Checking if the breaker has won requires a depth-first-search through the whole graph which is slow. We could fix that by keeping maker and breaker representation, but this would also double the move time. -->
My python env is too slow to run any MCTS on interesting problems. Although this is a research project and not production code, I don't think I can get around spending a lot of time optimizing.

## Things are slow!
Experiment:
+ 128 MCTS are run for 200 Iterations each on 8x8 hex graph
+ Batched processing 128 game states at a time on GNN running on GPU
+ 50 hidden channels, 20 layers

copy graph 2.4307218852572987  
hash 12.76779791075387  
process results 2.0024678500485606  
make move 25.61043147117016  
convert graph 19.889182167840772  
nn predictions 2.938752555011888  

+ hash time is to give unique hash to each graph in MCTS. Can be optimized fairly easily, I think. If not, it can be completely removed with different design of MCTS.
+ process results: This is to update the values in mcts. Although my env is not involved here, this takes almost as long as nn prediction.
+ Time to make moves in env is too slow.
  - Can be cut to ~ 1/4 if no dead and captured analysis, but that increases game lengths and average graph sizes by a lot.
	- The generality of our graph representations comes biting us back here. Using the graph representation, I run a local neighborhood search and use general rules to find dead or captured nodes. In a board representation, it is much harder to check if general rules are fulfilled that make squares dead or captured. However, in Hex, there are only so many local templates that render nodes dead or captured. Thus, given a bitwise board representation, one can fill dead and captured nodes with a few bitwise and/or's which is a lot faster than neighborhood search in the graph.
		* So I guess it would be smart and fast to keep a bitwise board representation together with the graph and use template matching. However, that is also somewhat sad, because we loose generalization to shannon-node-switchig game.
	- Maybe my graph filtering approach is also at fault here. I need to really think about the underlying Boost Graphing Library structure and how to use it optimally. Maybe removing vertices will be faster than filtering out.
+ Convert graph: Converting from graph-tool representation of my game to representation that my pytorch-geometric model can process takes far too long. The reason for this is that because I currently use graph-filtering, the vertex indices in graph-tool are not continuous and I have to reindex all edge indices. I could and probably should delete vertices instead of filtering out. However this makes some other things more difficult such as tracking the correspondence between vertices and board states for visualization.
+ NN prediction: In total these are 200 GNN predictions, with batches of 128 graphs each. Comparably very fast. We should aim to make this the bottleneck.

https://github.com/bhansconnect/fast-alphazero-general show how alpha-zero can be done with multiprocessing in python. However, it will be more tricky with GNNs, based on the way batching works in pytorch-geometric. Also, even with 8 or 16 cpu threads, running the env will still be the major bottleneck.

+ We can reduce the amount of required make\_moves in MCTS significantly by caching the graphs at each node. However this requires us to copy a lot of graphs, which is also expensive.
+ Graph-tool includes a information on how to extend it's c++ backend https://graph-tool.skewed.de/static/doc/demos/cppextensions/cppextensions.html. It may be possible to write the make\_move and remove\_dead\_and\_captured function in c++ while keeping the rest of the code in python.

There is a lot to optimize before we can run any meaningful experiments. Also, I think there is no way around cpu multiprocessing in the end. There are a few things I can think of optimizing in the current python implementation, but it might also make sense to implement some parts in C++.

## C++ to the rescue
I reimplemented my env in C++. Speed to run ten 11x11 games to completion with remove dead and captured, checking if game is won each move, takes 0.05 seconds in C++ and 0.35 seconds in python. Without remove dead and captured it is 0.025 seconds in C++ and 0.19 seconds in python.

Additionally, because I remove vertices instead of filtering them out in my C++ implementation, the graphs are already in a format that can be processed by pytorch\_geometric.

Thus, we cut the make-move time by a factor of 7 and the convert graph time should drop to zero. With a more sensible MCTS implementation we won't need hashing.

## Path forward and CrazyAra
+ I spend last days looking at CrazyAra code for better understanding of multi-thread MCTS in C++.
+ I could use that as an orientation to build my own multithread MCTS.
	- Then I could either implement GNN in C++, or use pybind and handle GNN stuff in python and pytorch-geometric.
+ However, maybe there is a better path making use of CrazyAra code: Fork CrazyAra, implement a state interface for my HexGraphs env. Then modify CrazyAra NN-Api to support GNNs and HexGraphs.
	- This seems like the most promising way to have a chance of beating top Hex AI.
	- I could use Graph Isomorphisms to make use of Monte-Carlo graph search. However, Weisfeiler-Lehman hashing to detect Isomorphisms might be too slow.
	- GNN framework to use? Pytorch-geometric is python-only. According to author, it should be possible to reimplement GNNs in C++ using torch-scatter c++ api. Or maybe use TorchScript. Or a completely different framework?
		+ Breaking news is that I tried TorchScript to transfer torch model from python to C++, but that seems to be hopeless as TorchScript does not support torch-scatter. Also, I failed to build torch-scatter in c++ with gpu support. Maybe I can make it work with torch-scatter some day (reimplementing GraphSAGE in C++ using torch-scatter and libtorch), but I'm also looking forward to alternatives.
		+ Claim by pytorch-geometric developer: *If your model does not depend on scatter_max and pooling algorithms, you should be able to reimplement your model with scatter_add and gather operations quite easily in C++ using libtorch.* So I guess I should try this.

# Mostly solved questions
+ Why do multithreading on single MCTS with virtual loss? Why not run n-threads MCTS in parallel instead and let each thread handle only one MCTS?
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
	- Currently the first (python) attempt is only faster for very deep MCTS trees. Otherwise second is faster.

+ GNN:
	- My input features to the GNN: Currently, 3D, one dim for *is neighbor of termial node 1*, one dim for *is neighbor of terminal node 2* and one dim for *Is it makers turn*. Should I add more? E.g. degree? Is it sensible to add is\_makers\_turn as an additional feature dimension to all nodes or is there a smarter way?

## Additional Notes:
+ I fixed a bug that resulted in the dead and captured algorithm not finding some captured cells for maker, while it found them for breaker. This does not change the theoretical value of the game in any position, but can make it easier to play for breaker and this might explain why we found a better breaker winrate in the DQN experiment.
