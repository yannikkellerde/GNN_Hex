# Using Graph Neural Networks to Improve Generalization in Self-Play Reinforcement Learning
## Reproducing the results from the Paper
### Retraining the models
Pretrained models are given in the checkpoints folder in the Rainbow submodule (`GN0/RainbowDQN/Rainbow/checkpoints`). You can also train your own models. Enter the Rainbow submodule `GN0/RainbowDQN/Rainbow` and use the train script to train models. The following shows the parameter settings for the models from the paper.
* GNN-S `python train.py --batch_size=256 --buffer_size=260000 --training_frames=200000000 --cnn_mode=False --use_amp=False --hex_size=7 --norm=False --model_name=modern_two_headed --prune_exploratories=True --grow=False --burnin=20000 --prioritized_er=True --prioritized_er_beta0=0.6 --prioritized_er_time=0 --final_eps=0.05 --init_eps=0.12 --eps_decay_frames=100000 --lr=0.0004 --loss_fn=mse --parallel_envs=128 --n_step=2 --gamma=0.97 --wandb_tag=lower_gamma,two_step,cnn --noisy_dqn=False --use_wandb=True --hidden_channels=35 --noisy_dqn=False --num_layers=10 --num_head_layers=2`
* FCN-S `python train.py --batch_size=256 --buffer_size=260000 --training_frames=200000000 --cnn_mode=True --use_amp=False --hex_size=7 --norm=False --model_name=fully_conv --prune_exploratories=True --grow=False --burnin=20000 --prioritized_er=True --prioritized_er_beta0=0.6 --prioritized_er_time=0 --final_eps=0.05 --init_eps=0.12 --eps_decay_frames=100000 --lr=0.0004 --loss_fn=mse --parallel_envs=128 --n_step=2 --gamma=0.97 --wandb_tag=lower_gamma,two_step,cnn --noisy_dqn=False --use_wandb=True --cnn_hex_size=7 --cnn_body_filters=22 --num_layers=10`
* GNN-L `python train.py --batch_size=256 --buffer_size=260000 --training_frames=200000000 --cnn_mode=False --use_amp=False --hex_size=11 --norm=False --model_name=modern_two_headed --prune_exploratories=True --grow=False --burnin=20000 --prioritized_er=True --prioritized_er_beta0=0.6 --prioritized_er_time=0 --final_eps=0.05 --init_eps=0.12 --eps_decay_frames=100000 --lr=0.0004 --loss_fn=mse --parallel_envs=128 --n_step=2 --gamma=0.97 --wandb_tag=lower_gamma,two_step,gnn --noisy_dqn=False --use_wandb=True --num_layers=15 --hidden_channels=110 --num_head_layers=2`
* FCN-L `python train.py --batch_size=256 --buffer_size=260000 --training_frames=200000000 --cnn_mode=True --use_amp=False --hex_size=11 --norm=False --model_name=unet --prune_exploratories=True --grow=False --burnin=20000 --prioritized_er=True --prioritized_er_beta0=0.6 --prioritized_er_time=0 --final_eps=0.05 --init_eps=0.12 --eps_decay_frames=100000 --lr=0.0004 --loss_fn=mse --parallel_envs=128 --n_step=2 --gamma=0.97 --wandb_tag=lower_gamma,two_step,cnn --noisy_dqn=False --use_wandb=True --cnn_hex_size=11`
### Reproducing the long range dependency table
Enter `GN0/tests` and run `python test_models_on_long_range_task.py`.
### Reproducing the board size transfer winrate plot
Enter `GN0/util` and run `python plotting.py`. Can take up to 10 minutes. You get a progress bar and intermediate results. At the end, the created barplots are shown.

## Documentation
This repository is organized into the following folders:  
No code:
+ [images](images): Graphics, subplots that where included or not included in the thesis.
+ [markdown](markdown): Intermediate progress reports
+ [slides](slides): The mid-thesis powerpoint presentation
+ [docker](docker): Dockerfiles that create docker environments in which the scripts of this repositiory run.

Code:
+ [graph\_game](graph_game): The python graph game environments used by this project.
+ [GN0](GN0): Python algorithms and utilities for this project. E.g. RainbowDQN, elo evaluation, plotting
+ [rl\_loop](rl_loop): The python part of HexAra.
+ [cpp\_hex](cpp_hex): HexAra and the Hex/Shannon node-switching game c++ environment

The documentation of this repository is split into sub-readme files that summarize the contents for each significant subfolder. Additonally, key files include a short description at the top of the file.

All sub-readme:   
[/cpp\_hex/README.md](/cpp_hex/README.md)  
[/cpp\_hex/hex\_graph\_game/README.md](/cpp_hex/hex_graph_game/README.md)  
[/cpp\_hex/CrazyAra/README.md](/cpp_hex/CrazyAra/README.md)  
[/rl\_loop/README.md](/rl_loop/README.md)  
[/graph\_game/README.md](/graph_game/README.md)  
[/GN0/RainbowDQN/README.md](/GN0/RainbowDQN/README.md)  
[/GN0/util/README.md](/GN0/util/README.md)  
[/GN0/README.md](/GN0/README.md)  
[https://github.com/yannikkellerde/Rainbow/blob/main/README.md](https://github.com/yannikkellerde/Rainbow/blob/main/README.md)

## RainbowDQN
The RainbowDQN algorithm used in the thesis is forked from `schmidtdominik/Rainbow` and is in the following repository: [https://github.com/yannikkellerde/Rainbow](https://github.com/yannikkellerde/Rainbow). To make this project work with RainbowDQN, clone the repo into `GN0/RainbowDQN`. (Yes I know git submodules exist, but they confuse me :D)

## Usage
Installing graph-tool together with torch\_geometric can be tricky and lead to conflicts if done in wrong order, with wrong python versions or wrong package versions. I can only refer to the dockerfile [gt\_dockerfile](/docker/gt_dockerfile) or to the [environment.yml](environment.yml) which work at the time of writing.

### Running RainbowDQN
The entrypoint for RainbowDQN after you cloned the Rainbow repository into the correct location is `GN0/RainbowDQN/Rainbow/train.py`. An example training command to train a GNN on 11x11 Hex would be: `python train.py --batch_size=256 --buffer_size=260000 --training_frames=200000000 --cnn_mode=False --use_amp=False --hex_size=11 --norm=False --model_name=modern_two_headed --prune_exploratories=True --grow=False --burnin=20000 --prioritized_er=True --prioritized_er_beta0=0.6 --prioritized_er_time=0 --final_eps=0.05 --init_eps=0.12 --eps_decay_frames=100000 --lr=0.0004 --loss_fn=mse --parallel_envs=128 --n_step=2 --gamma=0.97 --wandb_tag=lower_gamma,two_step,gnn --noisy_dqn=False --use_wandb=True --num_layers=15 --hidden_channels=110 --num_head_layers=2`. All scripts use wandb for logging, so make sure you set up wandb on your machine.

### Running HexAra
1. Check [/docker/dockerfile](/docker/dockerfile) for how to install C++ dependencies and create HexAra binary. (Or just create a docker container from it.
2. Use the `rl_loop/trace_model.py` script to trace your desired model with torch\_script
3. Entrypoint for training is `rl_loop/__main__.py`. Usually you'd want multiple gpus, one for training and multiple for data generation. Training process are started with `python -m rl_loop.__main__ --trainer --device-id=0`. Generator processes just with `python -m rl_loop.__main__ --device-id=N` with N for the GPU device id. Good luck :)
