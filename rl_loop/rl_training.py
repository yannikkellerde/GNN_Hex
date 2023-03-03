"""
@file: rl_training.py
Created on 20.11.19
@project: CrazyAra
@author: queensgambit

Functionality for conducting a single NN update within the reinforcement learning loop
"""

import sys
import glob
import logging
import wandb
from pathlib import Path

sys.path.append("../../../")
from rl_loop.train_config import TrainConfig, TrainObjects
from rl_loop.dataset_loader import get_loader
from rl_loop.train_util import get_metrics
from rl_loop.trainer_agent_pytorch import TrainerAgentPytorch, load_torch_state, save_torch_state, get_context, export_as_script_module
from GN0.torch_script_models import get_current_model, Unet
import torch
from torch_geometric.loader import DataLoader
import os


def update_network(nn_update_idx, pt_filename, trace_torch, main_config, train_config: TrainConfig, model_contender_dir, model_name, in_memory_dataset=False, cnn_mode=False):
    """
    Creates a new NN checkpoint in the model contender directory after training using the game files stored in the
     training directory
    :param nn_update_idx: Defines how many updates of the nn has already been done. This index should be incremented
    after every update.
    :param symbol_filename: Architecture definition file
    :param params_filename: Weight file which will be loaded before training
    :param tar_filename: Filepath to the model for pytorch
    Updates the neural network with the newly acquired games from the replay memory
    :param convert_to_onnx: Boolean indicating if the network shall be exported to ONNX to allow TensorRT inference
    :param main_config: Dict of the main_config (imported from main_config.py)
    :param train_config: Dict of the train_config (imported from train_config.py)
    :param model_contender_dir: String of the contender directory path
    """

    # set the context on CPU, switch to GPU if there is one available (strongly recommended for training)
    ctx = torch.device("cuda") if train_config.context == "gpu" else torch.device("cpu")
    # set a specific seed value for reproducibility
    train_config.nb_parts = len([os.path.join(main_config["planes_train_dir"],d) for d in os.listdir(main_config["planes_train_dir"])])
    logging.info("number parts for training: %d" % train_config.nb_parts)
    train_objects = TrainObjects()

    if train_config.nb_parts <= 0:
        raise Exception('No .zip files for training available. Check the path in main_config["planes_train_dir"]:'
                        ' %s' % main_config["planes_train_dir"])

    val_data = get_loader(train_config, dataset_type="val",cnn_mode=cnn_mode)

    net = _get_net(ctx, train_config, pt_filename)

    train_objects.metrics = get_metrics(train_config)

    train_config.export_weights = True  # save intermediate results to handle spikes
    train_agent = TrainerAgentPytorch(net, val_data, train_config, train_objects, use_rtpt=False, in_memory_dataset=in_memory_dataset, cnn_mode=cnn_mode)

    (val_value_loss_final, val_policy_loss_final, val_value_acc_sign_final,
     val_policy_acc_final), _ = train_agent.train()
    prefix = os.path.join(model_contender_dir,"weights-%.5f-%.5f-%.3f-%.3f" % (val_value_loss_final, val_policy_loss_final,
                                                                   val_value_acc_sign_final, val_policy_acc_final))

    _export_net(trace_torch, net, prefix,
                train_config, model_contender_dir, model_name)
    return net


def _export_net(trace_torch, net, prefix,
                train_config, model_contender_dir, model_name):
    """
    Export function saves both the architecture and the weights and optionally saves it as onnx
    """
    net.eval()
    save_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.lr),
                     '%s-%04d.pt' % (prefix, wandb.run.step))
    if trace_torch:
        export_as_script_module(net,os.path.join(model_contender_dir,model_name+"_model.pt"))

def _get_net(ctx, train_config, pt_filename):
    """
    Loads the network object and weights.
    """
    if train_config.net_type == "unet":
        net = Unet(3)
    elif train_config.net_type == "cnn":
        net = get_current_model("PV_CNN")
    else:
        net=get_current_model(net_type=train_config.net_type,hidden_channels=train_config.hidden_channels,hidden_layers=train_config.hidden_layers,policy_layers=train_config.policy_layers,value_layers=train_config.value_layers,in_channels=train_config.in_channels,swap_allowed=train_config.swap_allowed,norm=train_config.norm)
    net.to(ctx)
    if pt_filename!="":
        load_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.lr), pt_filename, train_config.device_id)
    return net
