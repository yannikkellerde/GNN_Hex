# GNN models in C++
+ I managed to transfer my GNN from Python to C++ via TorchScript.
	- torch\_scatter and torch\_sparse have c++ and TorchScript support via custom operators.
	- Works, even with cuda :)
+ ONNX would likely also work if I register torch\_sparse and torch\_scatter custom ops myself for onnx.
	- However, I don't think that is of any use as other frameworks don't have something like torch\_scatter or torch\_sparse and thus I still can only use libtorch.

# Graph structure and convert\_graph
+ Last time I claimed that converting my graph to tensors suitable for my gnns should be easy and fast now in c++, as I don't use graph-filtering and don't have to reindex my edges.
+ However, my graph framework, Boost Graph Library (BGL), made this hard.
	- Edges are not stored as a single vector, that I'd just copy over to a tensor and be done.
	- Abstraction going wrong: BGL tries so hard to abstract from underlying representation, that I don't get how to access edges efficently at all.
	- Naive approaches using BGL's edge api are terribly slow. This would destroy all efficiency gains from quicker make\_move.
+ I got so fed up with BGL that I decided to create my own graph framework from scratch with edges and vertex properties stored in vectors that can easily be copied to tensors.
	- Turns out, that was a pretty good decision, as even the move time dropped significiantly because of this.
	- Play 10 11x11 hex games with random moves, converting the graph to tensor representation ready to use for gnn each time:
		* move time: 0.013 seconds
		* convert_graph time: 0.01 seconds
		* total time: 0.023 seconds
		* for reference: with BGL, make move took already 0.05 seconds and convert_graph time was around 0.3 seconds.
		* In total, we now have ~30x efficientcy gain with respect to python implementation

# Roadmap
- [x] Create fast Hex Graph envirionment
- [x] Fast converision from Hex Graph envirionment to tensor representation
- [x] Convert python GNN model to C++
- [ ] Implement batching for GNN tensors
- [ ] Integrate with MCTS from CrazyAra
