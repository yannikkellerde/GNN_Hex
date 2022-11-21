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
from pathlib import Path

sys.path.append("../../../")
from rl_loop.train_config import TrainConfig, TrainObjects
from rl_loop.dataset_loader import load_pgn_dataset,_get_loader
from DeepCrazyhouse.src.training.lr_schedules.lr_schedules import MomentumSchedule, LinearWarmUp,\
    CosineAnnealingSchedule
from rl_loop.train_util import get_metrics
from GN0.models import get_pre_defined
from rl_loop.trainer_agent_pytorch import TrainerAgentPytorch, load_torch_state, save_torch_state, get_context, export_as_script_module
import torch
from torch_geometric.loader import DataLoader
import os


def update_network(queue, nn_update_idx, pt_filename, trace_torch, main_config, train_config: TrainConfig, model_contender_dir, model_name):
    """
    Creates a new NN checkpoint in the model contender directory after training using the game files stored in the
     training directory
    :param queue: Queue object used to return items
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
    :return: k_steps_final
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

    val_data = _get_loader(train_config, dataset_type="val")

    # calculate how many iterations per epoch exist
    nb_it_per_epoch = len(val_data) * train_config.nb_parts
    # one iteration is defined by passing 1 batch and doing backprop
    train_config.total_it = int(nb_it_per_epoch * train_config.nb_training_epochs)

    train_objects.lr_schedule = CosineAnnealingSchedule(train_config.min_lr, train_config.max_lr, max(train_config.total_it * .7, 1))
    train_objects.lr_schedule = LinearWarmUp(train_objects.lr_schedule, start_lr=train_config.min_lr, length=max(train_config.total_it * .25, 1))
    train_objects.momentum_schedule = MomentumSchedule(train_objects.lr_schedule, train_config.min_lr, train_config.max_lr,
                                         train_config.min_momentum, train_config.max_momentum)

    net = _get_net(ctx, train_config, pt_filename)

    train_objects.metrics = get_metrics(train_config)

    train_config.export_weights = True  # save intermediate results to handle spikes
    train_agent = TrainerAgentPytorch(net, val_data, train_config, train_objects, use_rtpt=False)

    # iteration counter used for the momentum and learning rate schedule
    cur_it = train_config.k_steps_initial * train_config.batch_steps
    (k_steps_final, val_value_loss_final, val_policy_loss_final, val_value_acc_sign_final,
     val_policy_acc_final), (_, _) = train_agent.train(cur_it)
    prefix = os.path.join(model_contender_dir,"weights-%.5f-%.5f-%.3f-%.3f" % (val_value_loss_final, val_policy_loss_final,
                                                                   val_value_acc_sign_final, val_policy_acc_final))

    _export_net(trace_torch, k_steps_final, net, prefix,
                train_config, model_contender_dir, model_name)

    logging.info("k_steps_final %d" % k_steps_final)
    queue.put(k_steps_final)


def _export_net(trace_torch, k_steps_final, net, prefix,
                train_config, model_contender_dir, model_name):
    """
    Export function saves both the architecture and the weights and optionally saves it as onnx
    """
    net.eval()
    save_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.max_lr),
                     '%s-%04d.pt' % (prefix, k_steps_final))
    if trace_torch:
        export_as_script_module(net,os.path.join(model_contender_dir,model_name+"_model.pt"))

def _get_net(ctx, train_config, pt_filename):
    """
    Loads the network object and weights.
    """
    net = get_pre_defined("HexAra")
    net.to(ctx)
    if pt_filename!="":
        load_torch_state(net, torch.optim.SGD(net.parameters(), lr=train_config.max_lr), pt_filename, train_config.device_id)
    return net
