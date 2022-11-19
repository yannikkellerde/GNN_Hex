+ How does threading / batching / virtual loss work in CrazyAra?
	- Doe multiple threads batch results together for nn\_evaluation?
		* It does not seem like that to me, as each SearchThread is running it's own create_mini_batch() and runs inference on it.
		* This surprises me, as I would have expected the most efficiency gains from batching the data from multiple threads together.
	- `searchThreads[i]->set_root_node(rootNode.get());` -> So many threads are working on the same MCTS tree.
		* If I increase the batch size or the number of threads, this results in huge virtual loss counts. (For batch size 128 and num threads 8, virtual loss climbs up to 150).
		* Sometimes this results in `Assertion `d->virtualLossCounter[childIdx] != 0' failed` during revert_virtual_loss. Maybe you remember, why you put that assert there and how you fixed the problem?
	- During Selfplay for RL data generation, we need to run muliple games anyway. To me, it seems like this is done sequentially, one after the other. Why not run many games simultaniously and batch nn inputs from them together? This would avoid a lot of virutal loss and allow for higher batch sizes. 
	- What number of threads / batch sizes do you use when running on a single gpu?
	- How much virtual loss can one MCTS handle without biasing the tree search too much?

+ Why is the default option in `options.cpp` and `rl\_config.py` to use only batch size 8. This seems very inefficient to me, as my (and I assume most) NNs/GPUs are able to process much larger batches in parallel. Is this just a bad default, or is it to avoid the problems with virtual loss, that I am experiencing.
+ Where can I find sensible defaults for RL training with CrazyAra. Are the values in rl\_config.py or options.cpp sensible defaults
	- Or which parameters where used in other experiments.
+ Why aren't MCTS search trees reused by default for RL in crazyara options?
+ What is the idea of wdl and ply loss?
