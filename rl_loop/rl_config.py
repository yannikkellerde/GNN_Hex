"""
@file: rl_config.py
Created on 01.04.2021
@project: CrazyAra
@author: queensgambit, maxalexger

Configuration file for Reinforcement Learning
"""
from dataclasses import dataclass
import os


@dataclass
class RLConfig:
    """Dataclass storing the options (except UCI options) for executing reinforcement learning."""
    # How many arena games will be done to judge the quality of the new network
    arena_games: int = 100
    arena_threads: int = 6
    do_arena_eval: bool = False
    winrate_eval_freq: int = 7200 # seconds
    # Directory where the executable is located and where the selfplay data will be stored
    binary_dir: str = os.path.abspath(f'data/RL/')
    binary_name: str = f'HexAra'
    model_name: str = "torch_script"
    # How many times to train the NN, create a model contender or generate nn_update_files games
    nb_nn_updates: int = 10
    # How many new generated training files are needed to apply an update to the NN
    nb_selfplay_games_per_thread: int = 300
    selfplay_threads = 20
    nn_update_files: int = 1
    precision: str = f'float16'
    # Replay Memory
    rm_nb_files: int = 7  # how many data packages/files shall be randomly taken from memory
    rm_fraction_for_selection: float = 0.3  # which percentage of the most recent memory shall be taken into account
    # The UCI_Variant. Must be in ["3check", "atomic", "chess", "crazyhouse",
    # "giveaway" (= antichess), "horde", "kingofthehill", "racingkings"]
    uci_variant: str = 'hex'


@dataclass
class UCIConfig:
    """
    Dataclass which contains the UCI Options that are used during Reinforcement Learning.
    The options will be passed to the binary before game generation starts.
    """
    Swap_Allowed: bool = False
    Hex_Size: int = 11
    Allow_Early_Stopping: bool = False
    Batch_Size: int = 8
    Centi_Dirichlet_Alpha: int = 30  # default: 20
    Centi_Dirichlet_Epsilon: int = 25
    Centi_Epsilon_Checks: int = 0
    Centi_Epsilon_Greedy: int = 0
    Centi_Node_Temperature: int = 100
    Centi_Resign_Probability: int = 90
    Centi_Q_Value_Weight: int = 0
    Centi_Quick_Probability: int = 0
    Centi_Temperature: int = 170    # Originally 80
    EPD_File_Path: str = "<empty>"
    MaxInitPly: int = 0  # default: 30
    MCTS_Solver: bool = True
    MeanInitPly: int = 0  # default: 15
    Milli_Policy_Clip_Thresh: int = 10
    Nodes: int = 800
    Reuse_Tree: str = False
    Search_Type: str = f'mcts'
    Selfplay_Chunk_Size: int = 128  # default: 128
    Selfplay_Number_Chunks: int = 640  # default: 640
    Simulations: int = 3200
    Temperature_Moves: int = 500  # CZ: 500
    Centi_Temperature_Decay: int = 99  # CZ: 500
    Timeout_MS: int = 0
    UCI_Chess960: bool = False


@dataclass
class UCIConfigArena:
    """
    This class overrides the UCI options from the UCIConfig class for the arena tournament.
    All other options will be taken from the UCIConfig class.
    """
    Centi_Temperature: int = 60
    Temperature_Moves: int = 10


