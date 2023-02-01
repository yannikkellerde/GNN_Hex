# Lessons from training supervised on HexAra data
+ Let's talk about losses
	- Question: When training a classification task, one should use cross-entropy loss. However, MSE usually also works. For high-dim discrete classfication tasks, cross-entropy loss is usually higher than MSE by factor of ~ 1000. So the loss function for cross-entropy is steeper. So the gradients with respect to the loss of each parameter of the net are larger. So to get the same rate of parameter update when switching from cross-entropy to MSE one would have to increase lr by factor of 1000? I've never heard of this and it sounds wrong to me, but what is the error in that logic?
	- For HexAra, I jointly update value head and policy head by adding theire respective losses. Policy loss is cross-entropy and value loss is MSE. Policy loss is always ~ factor 1000 higher than value loss. I wanted the net to care the same about policy and value. So I scaled the policy and value loss to be about the same. This was probably stupid, because only by removing that scaling, the network started to learn again.
	- Should I use BCE loss for value instead of MSE?

+ Initialization problems in GNN
	- Just trying to optimize the value loss does not work. The GNN body (GraphSAGE block) does not seem to start outputting significantly different aggregated embeddings for different positions.
	- First training the GNN with only policy loss does work and using that model to then only train with value loss does also work.
	- Too deep GNNs fail to start learning anything, even with combined loss.
		* ~ 15 layers seem to be the limit. Could be enough to train a good model, but still weird that it would not start learning at all for deeper models.
		* 15 layers, 60 hidden dim is at least well enough to completely overfit to 3000 mohex games.
	- Currently, I am just letting torch\_geometric and torch handle initialization automatically. Is there anything better I can do?
	- It has to be initialization problem, because training large GNNs works in RainbowDQN when growing from smaller models.
+ I found out that torch\_geometric offers two modes for LayerNorm. Node-level and Graph-level. Graph level seems to really hurt the training process. Node level I am not sure. Might actually be benificial.

+ Some experiments
	- Performance limit of RainbowDQN was not reached with previous model. Running it a little longer with an improved model architecture results in signficiantly stronger model [https://wandb.ai/yannikkellerde/rainbow_hex/runs/rc0hl84y?workspace=user-yannikkellerde](https://wandb.ai/yannikkellerde/rainbow_hex/runs/rc0hl84y?workspace=user-yannikkellerde)
		* This new model actually performs a lot better in the experiment (Than the previous model) vs mohex 0.5s. mohex 134 vs RainbowDQN 130
	- Success with imitation learning from mohex 0.5s. [https://wandb.ai/yannikkellerde/HexAra/runs/ddaja3q0?workspace=user-yannikkellerde](https://wandb.ai/yannikkellerde/HexAra/runs/ddaja3q0?workspace=user-yannikkellerde).
		* Removed explicit swap learning. This seemed to make some problems. Can still get nice swap maps based on position value at second move.
		* ~0.5 policy acc and ~0.7 value acc sign seem to be performance limit. Not sure how much of this is due to mohex randomness and how much due to failure to imitate.
		* Now checking how this model performs against it's teacher and if it can outperform it's teacher if I add MCTS myself. Currently working on this.
