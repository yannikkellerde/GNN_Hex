# Some model evaluation utilities used for RainbowDQN

Clone [https://github.com/yannikkellerde/Rainbow](https://github.com/yannikkellerde/Rainbow) into this folder.

The files in this folder are:
+ [evaluate\_elo.py](evaluate_elo.py): Compute elos of RainbowDQN agents during training. Also used to evaluate the Hexara models.
+ [mohex\_communicator.py](mohex_communicator.py): API to make it possible to evaluate mohex 2.0 in [evaluate\_elo.py](evaluate_elo.py).
+ [visualize\_transitions.py](visualize_transitions.py): Ensure the correctness of transitions in the replay buffer by visualizing them.
