# Python algorithms for graph games

This folder implements all the python algorithms for game play on graph games.

The files in this folder are:
+ [models.py](models.py): This implements all models used in python in this thesis
+ [torch\_script\_models.py](torch_script_models.py): This implements the models to be traced by torch\_script and imported into C++
+ [unet\_parts.py](unet_parts.py): This file is from [https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet\_parts.py](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py) and implements components for the U-net

The thesis-relevant subfolders are:
+ [RainbowDQN](RainbowDQN): This is where my RainbowDQN fork is supposed to be cloned into [https://github.com/yannikkellerde/Rainbow](https://github.com/yannikkellerde/Rainbow)
+ [util](util): This contains key utility for using my graph games with torch\_geometrics graph neural networks.
+ [tests](tests): Files to test the algorithm parts for correctness

Other subfolders are failed/unfinished attempts at other algorithms:
+ [alpha\_zero](alpha_zero): An attempt at implementing alpha\_zero in python. Ultimately unfinished because of environment runtime problems.
+ [supervised](supervised): Some intial attempts at learning supervised tasks with GNNs on my graph-game. NOT the mohex imitation part.
+ [other](other): Unfinished attempts with Monte-Carlo Policy Iteration or some attempts of GNNs with tensorflow.
