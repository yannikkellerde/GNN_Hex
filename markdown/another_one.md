# GNN stuff
+ I did read the [Principal Neighborhood Aggregation](https://proceedings.neurips.cc/paper/2020/file/99cad265a1768cc2dd013f0e740300ae-Paper.pdf) paper
+ I created a new GNN using PNA building blocks from pytorch\_geometric.
	- min, max and mean aggregation + degree scaling
	- + More sophisticated architecture than GraphSage that performs better on benchmarks
	- + Higher expressive power
	- + Degree scaling fixes problem where the GNN could not decide between nodes based on amout of neighbors (if all neighbors are isomorph)
	- - Forward pass takes significantly longer (Back to nn eval being biggest bottleneck). Maybe I can get away with less layers, but unsure.
	- - Significantly higher gpu memory requirements. Training on home GPU is over. Luckily dgx gpus have enough memory.

# Training run on DGX
+ In contrast to [AlphaGo Zero](https://www.nature.com/articles/nature24270), [AlphaZero](https://arxiv.org/pdf/1712.01815.pdf) does always use the latest model for data generation instead of evaluating and only choosing new model based on winrate against previous.
+ I do that too now, running 1 training process and two data generation processes in parallel, always using the latest model for data generation. (3 GPU on DGX)
+ Just starting to get things working without erroring. Still to find out if this will work.

# Misc
+ I found [this](https://arxiv.org/abs/2107.08387) paper which is pretty close to what I do.
