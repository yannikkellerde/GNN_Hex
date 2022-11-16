## Current status
+ Combined Hex graphs, GNNs and MCTS from CrazyAra.
+ Can generate selfplay MCTS games using GNN in C++ binary and save generated training data to file.
+ Arena (Letting two nets play against each other) also works.
+ Implemented in (near future) unmergable way
	- Fully replaced NN-api and State-api with my own.

## What is missing
+ The main loop and NN training part.
	- CrazyAra implements this fully in python, using the C++ binary to generate training data and then loads it from file.
	- I think I'll go the same path

## What might be missing (Not sure if worth implementing yet)
+ Multithreading in CrazyAra seems to be built with the concept of one thread per GPU.
	- This does make sense for super fast environments
	- For my case, I might be able to profit from the fact that there are usually more CPU cores available than GPUs.

## Batching, virtual loss and speed
+ During MCTS, total required nn-prediction time gets lower with higher batch size (up to a point)
+ Thus, CrazyAra uses virtual loss to batch together multiple states (hex graphs in my case)
+ However, this has a limit, as too many virtual losses significantly biases the MCTS
	- This gets worse, the smaller the game (e.g. Hex 5x5)
	- Yet, for smaller games, we would profit even more from higher batch sizes (We can batch a lot more 5x5 Hex states together than 11x11 Hex)
	- CrazyAra has a default batch size of 8, probably not to bias the MCTS too much with virtual loss. However, this is highly inefficient in terms of prediction runtime (see below).
+ My Idea: During training data generation in the RL loop, we anyway want to selfplay many games at once. So why not play 256 games at once and run 256 MCTS at once (one for each game) and batch the nn-inputs from each of them together. 
	- Memory requirements of course would be higher to store all those MCTS trees, but it sounds like it should be managable to me.
	- I have not found any implementation that does that, so maybe I am missing something.
		* Probably I should ask Johannes about this.
+ I'd like to learn 5x5 Hex using CrazyAra first to make sure everything is working before trying to learn 11x11 Hex on DGX. With RainbowDQN, learning 5x5 hex perfectly took around 30 minutes on my home computer. However, I am somewhat worried that with the low batchsizes required, this might take significantly longer on 5x5. E.g. optimal would be to process ~2048 5x5 Hex graphs at once, but I am worried that even using batch size 8 introduces too high of a bias with virtual loss in 5x5 Hex.

## Performance of components during selfplay with MCTS on 11x11 Hex
#### Batch size 8
NN prediction is clearly biggest bottleneck

|      task      |  total time (μs) |
| -------------- | ---------------- |
|         collate|           1978118|
|   convert graph|          12123550|
|      file write|              2553|
|       make move|          12384798|
|      nn predict|          41036071|
|    save samples|              8517|
|           total|          76064880|


#### Batch size 64
NN prediction is not a big bottleneck and we manage to make a lot more moves (explore more MCTS nodes), which makes make move and convert graph the biggest bottlenecks. In this case, we would likely profit from multithreading.

|      task      |  total time (μs) |
| -------------- | ---------------- |
|         collate|           2713624|
|   convert graph|          20385749|
|      file write|              2595|
|       make move|          26283928|
|      nn predict|          14884322|
|    save samples|              9140|
|           total|          76858753|


## More Questions
+ What is the point of *mctsagentbatch.cpp*? (Author Jannis Blüml)
