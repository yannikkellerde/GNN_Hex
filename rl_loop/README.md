# The python part of HexAra training

Note: Most of the files in this folder are bootstrapped from [https://github.com/QueensGambit/CrazyAra](https://github.com/QueensGambit/CrazyAra)

The relevant files required for running HexAra are the following:
+ `__main__.py`: This is the entrypoint for HexAra and handles the loop for either training data generation or model training (depending on the arguments)
+ `rl_training.py`: Handle NN import/export (torch\_script) and call the `trainer_agent_pytorch.py`
+ `trainer_agent_pytorch.py`: Train the GNN from the loaded training dataset and evaluate on the validation set.
+ `dataset_loader.py`: Load generated HexAra training data into torch\_geometric datasets.
+ `binaryio.py`: Communicate with HexAra binary.
+ `fileio.py`: Manage generated training data in the filesystem.
+ `train_config.py`, `rl_config.py`, `main_config.py`: Various configurations and hyperparameters of the HexAra algorithm

Additional relevant files not present in CrazyAra include
+ `generate_mohex_data.py`: Generate supervised learning date using mohex. (For Figure 5.7)
+ `plotting.py`: Swapmaps and starting eval images
+ `trace_model.py`: Trace a python model using torch\_script so it can later be loaded via C++
+ `inspect_rl_data.py`: Visualize generated training data to ensure correctness
+ `model_binary_player.py`: Python API for the HexAra binary that makes comparison in `GN0/RainbowDQN/evaluate_elo.py` possible. Relevant for Table 5.2.
