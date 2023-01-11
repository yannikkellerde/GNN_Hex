"""
@file: dataset_loader.py
Created on 22.10.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""
import os
import logging
from rl_loop.main_config import main_config
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_pgn_dataset(
    dataset_type="train", part_id=0, verbose=True, q_value_ratio=0,
):
    """
    Loads one part of the pgn dataset in form of planes / multidimensional numpy array.
    It reads all files which are located either in the main_config['test_dir'] or main_config['test_dir']

    :param dataset_type: either ['train', 'test', 'mate_in_one']
    :param part_id: Decides which part of the data set will be loaded
    :param verbose: True if the log message shall be shown
    :param normalize: True if the inputs shall be normalized to 0-1
    ! Note this only supported for hist-length=1 at the moment
    :param q_value_ratio: Ratio for mixing the value return with the corresponding q-value
    For a ratio of 0 no q-value information will be used. Value must be in [0, 1]
    :return: torch tensors:
            start_indices - defines the index where each game starts
            node_features - the node representation for all games
            edge_indices - the edge indices for the graphs
            y_value - the game outcome (-1,0,1) for each board position
            y_policy - the movement policy for the next_move played
            plys_to_end - array of how many plys to the end of the game for each position.
             This can be used to apply discounting
    """

    if dataset_type == "train":
        data_folders = [os.path.join(main_config["planes_train_dir"],d) for d in os.listdir(main_config["planes_train_dir"])]
    elif dataset_type == "val":
        data_folders = [os.path.join(main_config["planes_val_dir"],d) for d in os.listdir(main_config["planes_val_dir"])]
    elif dataset_type == "test":
        data_folders = [os.path.join(main_config["planes_test_dir"],d) for d in os.listdir(main_config["planes_test_dir"])]
    else:
        raise Exception(
            'Invalid dataset type "%s" given. It must be either "train", "val", "test"' % dataset_type
        )

    data_folders.sort()

    if len(data_folders) < part_id + 1:
        raise Exception("There aren't enough parts available (%d parts) in the given directory for partid=%d"
                        % (len(data_folders), part_id))

    pgn_dataset = data_folders[part_id]
    if verbose:
        logging.debug("loading: %s ...", pgn_dataset)
        logging.debug("")

    files_tensor_list = ("node_features","edge_indices","policy")
    files_tensor = ("value","best_q")
    out = {}
    for fname in files_tensor:
        out[fname] = next(torch.jit.load(os.path.join(pgn_dataset,"torch",fname+".pt")).parameters())
    for fname in files_tensor_list:
        out[fname] = list(torch.jit.load(os.path.join(pgn_dataset,"torch",fname+".pt")).parameters())

    if q_value_ratio != 0:
        out["value"] = (1-q_value_ratio) * out["value"] + q_value_ratio * out["best_q"]

    data_list = []
    for i in range(len(out["node_features"])):
        data_list.append(Data(x=out["node_features"][i], edge_index=out["edge_indices"][i], y=out["value"][i], policy=out["policy"][i]))

    print(data_list[-1].y, data_list[-1].shape)
    print(sum([torch.sum(x.y) for x in data_list]),sum([len(x.y) for x in data_list]))

    return data_list


def _get_loader(train_config,part_id=0,dataset_type:str="val"):
    """
    Returns the validation loader.
    """
    data_list = load_pgn_dataset(dataset_type=dataset_type,
                                   part_id=part_id,
                                   verbose=False,
                                   q_value_ratio=train_config.q_value_ratio)
    data = DataLoader(data_list, shuffle=True, batch_size=train_config.batch_size,
                          num_workers=train_config.cpu_count)
    return data
