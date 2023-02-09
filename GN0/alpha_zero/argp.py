import argparse
import distutils
import os

def read_args():
    parse_bool = lambda b: bool(distutils.util.strtobool(b))
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cnn_mode",type=parse_bool,default=False)
    parser.add_argument("--hex_size",type=int,default=5)
    parser.add_argument("--capacity",type=int,default=2**16)
    parser.add_argument("--online_wandb",type=parse_bool,default=True)
    parser.add_argument("--baseline_network_path",type=str,default="/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/RainbowDQN/Rainbow/checkpoints/misty-firebrand-26/5/checkpoint_3200000.pt")
    parser.add_argument("--num_iterations",type=int,default=10)
    parser.add_argument("--num_episodes",type=int,default=10)
    parser.add_argument("--training_temp",type=float,default=1)
    parser.add_argument("--temp_threshold",type=int,default=2)
    parser.add_argument("--num_epochs",type=int,default=1000)
    parser.add_argument("--checkpoint",type=str,default="checkpoints")
    parser.add_argument("--required_beat_old_model_winrate",type=float,default=0.55)
    parser.add_argument("--hidden_channels",type=int,default=25)
    parser.add_argument("--num_layers",type=int,default=8)
    parser.add_argument("--head_layers",type=int,default=2)
    parser.add_argument("--num_training_epochs",type=int,default=3)
    parser.add_argument("--lr",type=int,default=0.00025)
    parser.add_argument("--training_batch_size",type=int,default=128)
    parser.add_argument("--mcts_batch_size",type=int,default=256) # Actual batch size about half, because split between maker and breaker
    parser.add_argument("--batched_mcts",type=parse_bool,default=True)
    parser.add_argument("--weight_decay",type=float,default=1e-5)
    parser.add_argument("--cpuct",type=float,default=1.0)

    args = parser.parse_args()
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    return args
