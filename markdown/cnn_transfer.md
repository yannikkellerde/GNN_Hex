# CNN transfer
+ First try: similar to AlphaGo arch, but no skip layers.
	- Just 3x3 conv layers with relu and no pooling, then value and advantage head with fully connected layers.
	- Then do transfer by padding 5x5 board with zeros during training. Fill that padding during 6x6.
	- Result: can learn 5x5, basically zero transfer to 6x6.
+ Second try: fully convolutional
	- Well, turns out most cnn architectures only require fixed input size, because they use a fully connected layer in the end.
	- If we go fully convolutional, scaling is actually easy.
		* I should have known this from computer vision, but first general consensus if you google is: CNN require fixed input size.
	- So I use fully convolutial architecture with single Q-head (no duelling DQN) that only has 3x3 conv layers with padding='same' + relu and finishes with 1x1 conv layer with one filter output. Then flatten.
	- Just fixed a bug yesterday (forgot to filter out illegal moves in q-target computation).
		* Bug-free (hopefully) experiments on 5x5 and 7x7 running since yesterday.
		* Results fresh from the oven
