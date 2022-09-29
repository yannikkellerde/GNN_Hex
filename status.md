# Using Graph Neural Networks to Improve Generalization in Self-Play Reinforcement Learning

## Hex implementation
- Fully implemented as *Shannon Node-Switching Game* -> Fully represented as Graph. Accompanying board representation optional, can be left out during training.
- Maker tries to connect two nodes in the graph by removing nodes and connecting neighbors, Breaker tries to prevent this by removing nodes without connecting the neighbors.
- Drawback: Without board representation it is not possible in polynomial time to swap colors (E.g. transform into an equivalent position where roles are reversed).
- Maker and Breaker solve fundamentally different problems and may need their own model.
- Some basic dead and captured node analysis implemented directly on the graph representation (No template matching)

## Hex agent trained with Rainbow DQN
- Why Rainbow DQN? Goal is still to train with MCTS/Alpha-zero method. However this may be very compute intesive (see below) and I just did a lot of Rainbow DQN in my RL course, so why not try it.
- Started with [this](https://github.com/schmidtdominik/Rainbow) implementation and transformed it to work with Graph-Nets and self-play
- Collect transitions into replay buffer via self-play. E.g. one-step next state is state after agent and opponent made a move.
### Graph Net Architecture
- [GraphSage](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv) which has some ResNet style properties.
- One shared body of 15 SageConv layers with 60 hidden layers.
	+ One head for breaker and maker each with two SageConv layers and two final linear layers for value and advantage (Duelling DQN).
### Rainbow DQN Features
- [x] Duelling DQN
- [x] Double DQN
- [x] 2-step returns
- [x] Prioritized Replay buffer
- [ ] Noisy Nets (Did not manage to make it work well)
- [ ] Distributional RL (too complicated)
### Training
- On my GTX 1070ti at home
- Did some curriculum training -> start with small network and hex size 5\*5. Then grow network and hex-size simultaiously
	+ Not sure if it helped, but didn't seem to hurt and I wanted to try it because I did Martin Mundt's continual ML course last semester.
	+ Does make some sense because smaller hex size is subproblem of larger hex size and has higher reward density.
- One Replay buffer for maker, one for breaker. Seperate training (but shared parameters in NN body).
- Required a lot of training time. Kept improving after four days of training.

## Results and observations

### Agent performance
- Slightly stronger than me thinking for ~10 seconds per move. (Agent takes only miliseconds, as it is not doing any search)
- Stronger than the [android MoHex version](https://play.google.com/store/apps/details?id=com.game.hex&hl=en&gl=US) that thinks for ~3 seconds per move on my smartphone.
- Weaker than MoHex 2.0 on my laptop thinking for ~10s per move.
- Weaker than [https://cleeff.github.io/hex/](https://cleeff.github.io/hex/) which also does no search and moves amost instantly.

### Observations
- Learning the breaker seems to be a significantly easier than learning the maker. Winrates averaged around 70% for breaker in all my runs (with random starting player and move). Exception is 5x5 where the agent fully solves all starting moves after a while.
- Losses stop sinking fairly early, while the agent strength keeps increasing.
- Transfer between hex sizes works great. An agent trained on the 11x11 board playing on 8x8 is almost as good as an agent trained on 8x8. An agent trained on 8x8 playing on 11x11 is not too bad at all and shows clear signs of upward transfer.

## The state of computer hex right now
- Best classical engine is MoHex 2.0 which uses MCTS (Without full rollouts and no Neural Networks).
- Best Hex AI of all time is probably [https://github.com/richemslie/galvanise\_zero](https://github.com/richemslie/galvanise_zero)
	+ Alpha Zero style training.
	+ Used 3 GTX 1080 for training ResNets with convolutional layers.
	+ Combines Python and C++.
	+ General game player that can tackle many games written in GDL.
- [This](https://github.com/harbecke/HexHex) one is very interesting. It's online agent [https://cleeff.github.io/hex/](https://cleeff.github.io/hex/) seems very strong.
	+ I am not 100% sure what their training Method is.
	+ On their github README they claim to use a similar training method and network architectures as Alpha Zero, but no MCTS, because it is *prohibitively expensive*.
	+ Not sure if I fully understand their code, but to me it seems just like Monte-Carlo Q Iteration with self play and CNN function approximation.
	+ I wrote the author and got an inconclusive answer.
	+ Very surprising if they were able to train such a strong model with such a "naive" algorithm as Monte-Carlo Q Iteration.
- I have not found anyone using Graph-Neural networks in a similar way as I do in self-play RL.

## Open Questions
### Graph Neural Networks
- How deep can/should I go e.g. can 15 layers be enough or should I go a lot deeper (there are examples of 1000 layer gnns out there).
- What should be the hidden channels vs num layers ratio for GNNs?
- Do GNNs profit from a pyramid style layout? E.g. less hidden channels each layer.
- How to do normalization for GNNs in an RL setting?
	+ In my Bachelors-thesis I did some continual RL and found layer-normalization to do wonders for my RL agent.
	+ I tried adding [GraphNorm](https://arxiv.org/abs/2009.03294) to my GNN.
		* In a preliminary experiment, where I just tried to learn Shannons electrical circuit for hex graphs (a supervised learning task), adding GraphNorm improved convergence quite a lot.
		* In the RL setting, it seemed to hurt performance more than it helped.
		* I think the reason is similar to why BatchNorm is usually not used in RL. GraphNorm also uses Batch statistics and they tend to change over time in the RL setting.


## Where do we go from here?
- I don't want to spend too much more time tuning Rainbow DQN hyperparameters, so I think it makes sense to move towards MCTS for training soon.
- Collecting MCTS transitions is a lot more expensive as we need to build tree each time. [https://github.com/harbecke/HexHex](https://github.com/harbecke/HexHex), which is purely written in python even calls it "prohibitively expensive".
- I am not sure if the bottleneck will be GPU or single thread CPU throughput. If single thread CPU throughput is a bottleneck, we need multiprocessing.
	+ Multiprocessing is a pain in python
	+ It might make sense to reimplement in C++
		* I can't use any of the existing C++ hex implementations, as they all work on the hex board representation and converting from board to graph is expensive.
		* I already use graph-tool in python which uses the boost graphing library as a backend, so translating into c++ with boost graphing library should not be too hard.
		* Torch has (beta) C++ api.
		* torch_geometric does not have a C++ api.
		* torch_scatter, which is the essential module on which torch geometric is based on has a C++ api.
		* With access to torch_scatter it should not be too hard to reimplement required torch_geometric classes in C++.
		* There are a lot of Alpha-Zero implementations in C++ out there, which I could repurpose. (e.g. open_spiel)
- Some alternative path would be to make [https://github.com/harbecke/HexHex](https://github.com/harbecke/HexHex) work with Graph Neural Networks to get a direct comparison between ResNet CNNs and Graph Neural networks for RL on Hex.

## Misc
- I tried enhancing my agent trained with DQN with alpha-zero style MCTS at test time. However this did not seem to increase strength. Either my implementation is wrong or treating the softmaxed advantages as the prior for MCTS is not a smart idea.
