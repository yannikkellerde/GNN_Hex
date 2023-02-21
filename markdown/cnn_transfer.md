# CNN transfer
+ First try: similar to AlphaGo arch, but no skip layers.
	- Just 3x3 conv layers with relu and no pooling, then value and advantage head with fully connected layers.
	- Then do transfer by padding 5x5 board with zeros during training. Fill that padding during 6x6.
	- Result: can learn 5x5, basically zero transfer to 6x6.
		* Also worse that gnn at learning 5x5

+ Second try: fully convolutional
	- Well, turns out most cnn architectures only require fixed input size, because they use a fully connected layer in the end.
	- If we go fully convolutional, scaling is actually easy.
		* I should have known this from computer vision, but I was stupid enough to just believe https://arxiv.org/pdf/2107.08387.pdf and the first google results.
	- So I use fully convolutial architecture with single Q-head (no duelling DQN) that only has 3x3 conv layers with padding='same' + relu and finishes with 1x1 conv layer with one filter output. Then flatten.
		* Inputs: 2 layers red, blue, 1 layer onturn.
		* Results: CNN beats GNN close to similar board size after training on 7x7, but GNN wins by a lot for transfer to very different board sizes (see results section)

+ Additional experiment: long range dependencies for CNN and GNN.
	- Use GNN and CNN trained on 7x7 to try and solve long range dependency problems on various board sizes
		* Result: GNN does a lot better, even on board sizes where CNN wins more games. (see results section)

+ What to make of this?
	- Even with this simple fully convolutional architecture (No pooling, no norm, no skip connections), CNN beats GNN on 7x7 Hex.
		* Not that surprising, main claim for GNN advantage was long range dependencies. 7x7 is somewhat to small for long range dependencies.
		* Maybe run another experiment training on 11x11 or 13x13? Need another CNN arch for that, current approach does not really scale.
	- Transfer advantge of GNN shown, but only to very different board sizes. On 8x8 and 9x9, CNN still has advantage.
	- Long range dependency problem does indeed show GNN advantage. For a high winrate, local patters do have also a significant impact however.

## Motivation in Methods?
+ I spend some paragraphs trying to motivate RainbowDQN and HexAra (AlphaZero) as approaches to use with GNNs.
+ Currently, I put them into Methods. Is that the place it belongs? Or should methods just be: *What did I do?*


## So the answers to my research questions are currently:
+ Can GNNs capture the relational structure of Hex better than GNNs?
	- There are concepts in hex that are captured better with GNNs in the graph structure.
		* E.g. long range dependencies such as in section above.
	- However, on 7x7, local patterns seem to be more important than long-range dependencies and CNN perform better.
+ Can GNNs aid knowledge transfer between board sizes in Hex?
	- In agreement with https://arxiv.org/pdf/2107.08387.pdf, I found that GNNs even retain knowledge, transfering to much larger board sizes. Fully conv CNN work better for transfer to slightly different board sizes, but catastrophically fail for transfer to much larger board sizes.
+ Which self-play reinforcement learning approaches work the best with GNNs?
	- Unclear, but I had more difficulties with AlphaZero MCTS method.
	- To get some reasonable results for the thesis paper, I should probably spend the time to implement CNN for HexAra too.
