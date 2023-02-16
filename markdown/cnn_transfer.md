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
	- Fresh results with similar training time for gnn and cnn
		* On 5x5: {'gnn': 15, 'cnn_5x5': 15}
		* Transfer to 6x6: {'cnn_5x5': 22, 'gnn': 20}
		* On 7x7: {'gnn_7x7': 24, 'cnn_7x7': 32}
		* Transfer to 8x8: {'gnn_7x7': 23, 'cnn_7x7': 49}
		
+ **I should not have believed this random arxiv paper [https://arxiv.org/pdf/2107.08387.pdf](https://arxiv.org/pdf/2107.08387.pdf). There is no benefit of GNNs for transfer and frankly I'm not sure if it is anywhere.**
+ .
+ .
+ .
+ Well, at this is the point in research where you question everything you did.
+ Lessons:
	- Do comparisons earlier.
	- Focus on one experiment first (Spending that time on HexAra was nice to learn C++ and stuff, but kind of wasted).
	- Be more sceptical of random arxiv papers.
+ I guess as this is a masters thesis and not a phd, the fact that you fail to show the thing you wanted to show does not immediately mean the thesis is a fail (if I understood it correctly)
	- Then I guess the question becomes how to write it in the thesis...

## The one case where GNNs do have an advantage
+ The long range dependency example from the introduction
[](/images/long_range_compare_positive.svg) [](/images/long_range_compare_negative.svg)
+ Depending on the position, one should or should not play top right as blue.
	- Tested with cnn trained on 7x7 and gnn trained on 7x7 with same kind of position on hex sizes 5-13.
	- CNN starts making first mistake on 7x7 and in total gets 20/32 correct.
	- GNN only starts making mistakes on 12x12 onwards and gets 29/32 correct.
