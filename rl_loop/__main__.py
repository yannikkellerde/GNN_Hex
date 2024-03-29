"""
@file: rl_loop.py
Created on 12.10.19
@project: crazy_ara
@author: queensgambit

Main reinforcement learning for generating games and train the neural network.
"""

import json
import distutils
import os
import sys
import logging
import argparse
from rtpt import RTPT
import dataclasses
from multiprocessing import Process, Queue
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt

from rl_loop.fileio import FileIO
from rl_loop.rl_utils import enable_logging, get_log_filename, get_current_binary_name, \
    extract_nn_update_idx_from_binary_name, change_binary_name
from rl_loop.binaryio import BinaryIO
from rl_loop.main_config import main_config
from rl_loop.train_config import TrainConfig
from rl_loop.rl_config import RLConfig, UCIConfigArena
from rl_loop.rl_training import update_network, _get_net
from rl_loop.plotting import show_eval_from_file,compute_and_plot_starting_eval,compute_and_plot_swapmap
from rl_loop.trainer_agent_pytorch import get_context
import torch, wandb
import time


class RLLoop:
    """
    This class uses the C++ binary to generate games and updates the network from the newly acquired games
    """

    def __init__(self, args, rl_config, nb_arena_games=100, lr_reduction=0.0001):
        """
        Constructor
        :param args: Command line arguments, see parse_args() for details.
        :param nb_arena_games: Number of games which will be generated in a tournament setting in order to determine
        :param lr_reduction: Learning rate reduction of maximum learning rate after a single NN update
        if the updated NN weights are stronger than the old one and by how much.
        be written to the new model filenames to track how many iterations the model has trained in total)
        """
        self.net = None
        self.arena_start = False # Set to true to continue a run that failed at arena stage
        self.args = args
        self.tc = TrainConfig()

        self.rl_config = rl_config

        self.file_io = FileIO(orig_binary_name=self.rl_config.binary_name, binary_dir=self.rl_config.binary_dir, uci_variant=self.rl_config.uci_variant, framework=self.tc.framework, model_name=self.rl_config.model_name,device_id=args.device_id)
        if args.trainer:
            ctx = torch.device("cuda") if self.tc.context == "gpu" else torch.device("cpu")
            self.starting_net = _get_net(ctx, self.tc, self.file_io.get_current_model_pt_file(),hex_size=self.args.hex_size)
        self.binary_io = None

        if nb_arena_games % 2 == 1:
            raise IOError(f'Number of games should be even to avoid giving a player an advantage')
        self.nb_arena_games = nb_arena_games
        self.lr_reduction = lr_reduction
        self.device_name = f'{args.context}_{args.device_id}'
        self.model_name = ""  # will be set in initialize()
        self.did_contender_win = True
        self.next_winrate_eval = time.time()

        # change working directory (otherwise binary would generate .zip files at .py location)
        os.chdir(self.file_io.binary_dir)
        self.tc.cwd = self.file_io.binary_dir
        logpath = os.path.join(self.tc.export_dir,"wandb_logs",str(self.args.device_id))
        os.makedirs(logpath,exist_ok=True)
        if self.args.continue_runs and os.path.exists("stored_ids.json"):
            with open("stored_ids.json","r") as f:
                id_map = json.load(f)
        else:
            id_map = {"0":wandb.util.generate_id(),"1":wandb.util.generate_id(),"2":wandb.util.generate_id()}
            with open("stored_ids.json","w") as f:
                json.dump(id_map,f)
        name_map = {0:"trainer",1:"evaluater",2:"generator"}
        group = "mohex+alpha0"
        wandb.init(resume='allow',id=id_map[str(self.args.device_id)],project='HexAra', save_code=True, config=dict(**rl_config.__dict__, **self.tc.__dict__, log_version=100),entity="yannikkellerde", mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=[], dir=logpath, job_type=name_map[int(self.args.device_id)],group=group)
        wandb.run.name = name_map[int(self.args.device_id)]
        # wandb.init(resume="must",id="19wl47yk",project='HexAra', save_code=True, config=dict(**rl_config.__dict__, **self.tc.__dict__, log_version=100),entity="yannikkellerde", mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=[], dir=os.path.join(self.tc.export_dir,"logs"))

        # The original binary name in TrainConfig will always stay the same & be a substring of the updated name
        self.current_binary_name = get_current_binary_name(self.file_io.binary_dir, self.rl_config.binary_name)

        self.nn_update_index = args.nn_update_idx
        if not args.trainer:  # only trainer gpu needs the update index as cmd line argument
            self.nn_update_index = extract_nn_update_idx_from_binary_name(self.current_binary_name)
        self.last_nn_update_index = self.nn_update_index + self.rl_config.nb_nn_updates

        # Continuously update the process name
        self.rtpt = RTPT(name_initials=self.tc.name_initials,
                         experiment_name=f'{self.rl_config.binary_name}_{self.rl_config.uci_variant}',
                         max_iterations=self.rl_config.nb_nn_updates, moving_avg_window_size=1)
        self.rtpt.start()
        if self.current_binary_name == "HexAra":
            self.current_binary_name = change_binary_name(self.file_io.binary_dir, self.current_binary_name,
                                                          self.rtpt._get_title(), self.nn_update_index)

    def initialize(self, is_arena=False):
        """
        Initializes the CrazyAra binary and loads the neural network weights
        is_arena: Signals that UCI option should be set for arena comparison
        :return:
        """
        if self.binary_io is not None:
            self.binary_io.stop_process(True)
        self.model_name = self.file_io.get_current_model_pt_file()
        self.binary_io = BinaryIO(binary_path=self.file_io.binary_dir+self.current_binary_name)
        self.binary_io.set_uci_options(self.rl_config.uci_variant, self.args.context, self.args.device_id, self.rl_config.precision, self.file_io.model_dir, self.file_io.model_contender_dir if self.rl_config.do_arena_eval else self.file_io.eval_checkpoint_dir, self.rl_config.selfplay_threads, self.rl_config.model_name, is_arena, cnn_mode=self.args.cnn_mode, hex_size=self.args.hex_size)
        self.binary_io.load_network()

    def check_for_new_model(self):
        """
        Checks if the current neural network generator has been updated and restarts the executable if this is the case
        :return:
        """

        new_binary_name = get_current_binary_name(self.file_io.binary_dir, self.rl_config.binary_name)
        if new_binary_name != self.current_binary_name:
            self.current_binary_name = new_binary_name
            # when binary name changes, also epoch changes
            self.nn_update_index = extract_nn_update_idx_from_binary_name(self.current_binary_name)

            # If a new model is available, the binary name has also changed
            # model_name = self.file_io.get_current_model_weight_file()
            # if model_name != "" and model_name != self.model_name:
            #     logging.info("Loading new model: %s" % model_name)
            self.rtpt.step()
        self.initialize()

    def evaluate(self):
        if self.file_io.is_there_checkpoint():
            self.initialize(is_arena=True)
            logging.info(f'Start arena tournament ({self.nb_arena_games} rounds)')
            self.did_contender_win, winrate = self.binary_io.compare_new_weights(self.nb_arena_games, self.rl_config.arena_threads)
            logs = dict(winrate=winrate);
            if self.did_contender_win:
                self.file_io.store_arena_pgn(wandb.run.step+1)
            wandb.log(logs)
            print("logging winrate",logs)
        self.file_io.copy_model_to_eval_checkpoint()
        self.next_winrate_eval = time.time()+self.rl_config.winrate_eval_freq


    def check_for_enough_train_data(self, number_files_to_update):
        """
        Checks if enough training games have been generated to trigger training a new network
        :param number_files_to_update: Number of newly generated files needed to trigger a new NN update
        :return: True, if enough training data was availble and a training run has been executed.
        """
        print("new generated:",self.file_io.get_number_generated_files(),"\ntotal available:",self.file_io.get_total_available_training_files(),"\nrequired:",number_files_to_update)
        if self.file_io.get_total_available_training_files() >= number_files_to_update or self.arena_start:
            try:
                del self.starting_net
            except:
                print("starting net already gone")
            if not self.arena_start:
                self.file_io.prepare_data_for_training(self.rl_config.rm_nb_files, self.rl_config.rm_fraction_for_selection,self.did_contender_win)
                self.tc.device_id = self.args.device_id
                logging.info("Start Training")
                self.net = update_network(self.nn_update_index,self.file_io.get_current_model_pt_file(),not self.args.no_trace_torch,main_config,self.tc,self.file_io.model_contender_dir,self.file_io.model_name, in_memory_dataset=self.args.in_memory_dataset,cnn_mode=self.args.cnn_mode, gao_mode=self.args.gao_mode, hex_size=self.args.hex_size)

                self.file_io.move_training_logs(self.nn_update_index)

                self.nn_update_index += 1
            self.arena_start = False

            if self.rl_config.do_arena_eval:
                self.initialize()
                logging.info(f'Start arena tournament ({self.nb_arena_games} rounds)')
                self.did_contender_win, winrate = self.binary_io.compare_new_weights(self.nb_arena_games, self.rl_config.arena_threads)
                logs = dict(winrate=winrate);
                if self.did_contender_win:
                    self.file_io.store_arena_pgn(wandb.run.step+1)
            else:
                logs = dict()
                self.did_contender_win = True
            if self.did_contender_win:
                logging.info("REPLACING current generator with contender")
                self.file_io.replace_current_model_with_contender()
            else:
                logging.info("KEEPING current generator")

            self.file_io.remove_intermediate_weight_files()

            self.rtpt.step()  # BUG: process changes it's name 1 iteration too late, fix?
            self.current_binary_name = change_binary_name(self.file_io.binary_dir, self.current_binary_name,
                                                          self.rtpt._get_title(), self.nn_update_index)
            # self.initialize()
            # if self.did_contender_win:
            #     self.initialize()
            #     plt.cla()
            #     if self.args.cnn_mode:
            #         fig = compute_and_plot_starting_eval(self.args.hex_size,self.net,device=get_context(self.args.context,self.args.device_id))
            #         logs["starting_policy"] = wandb.Image(fig)
            #         fig = compute_and_plot_swapmap(self.args.hex_size,self.net,device=get_context(self.args.context,self.args.device_id))
            #         logs["swapmap"] = wandb.Image(fig)
            #     else:
            #         self.binary_io.generate_starting_eval_img()
            #         fig = show_eval_from_file("starting_eval.txt",colored="top3",fontsize=6)
            #         logs["starting_policy"] = wandb.Image(fig)
            #         fig = show_eval_from_file("swap_map.txt",colored=".5",fontsize=6)
            #         logs["swapmap"] = wandb.Image(fig)
            #    self.binary_io.stop_process()
            wandb.log(logs)
        else:
            time.sleep(10)



def parse_args(cmd_args: list):
    """
    Parses command-line argument and returns them as a dictionary object
    :param cmd_args: Command-line arguments (sys.argv[1:])
    :return: Parsed arguments as dictionary object
    """
    parse_bool = lambda b: bool(distutils.util.strtobool(b))
    parser = argparse.ArgumentParser(description='Reinforcement learning loop')

    parser.add_argument('--hex_size', type=int, default=11,
                        help='Hex size to train on')

    parser.add_argument('--cnn_mode', type=parse_bool, default=False,
                        help='Use with cnn')
    parser.add_argument('--gao_mode', type=parse_bool, default=False,
                        help='Use with cnn')
    parser.add_argument('--context', type=str, default="gpu",
                        help='Computational device context to use. Possible values ["cpu", "gpu"]. (default: gpu)')
    parser.add_argument("--device-id", type=int, default=0,
                        help="GPU index to use for selfplay generation and/or network training. (default: 0)")
    parser.add_argument("--trainer", default=False, action="store_true",
                        help="The given GPU index is used for training the neural network."
                             " The gpu trainer will stop generating games and update the network as soon as enough"
                             " training samples have been acquired.  (default: False)")
    parser.add_argument("--evaluater", default=False, action="store_true",
                        help="This process will do arena evals")
    parser.add_argument('--export-no-log', default=False, action="store_true",
                        help="By default the log messages are stored in {context}_{device}.log."
                             " If this parameter is enabled no log messages will be stored")
    parser.add_argument('--nn-update-idx', type=int, default=0,
                        help="Index of how many NN updates have been done so far."
                             " This will be used to label the NN weights (default: 0)")
    parser.add_argument('--no-trace-torch', default=False, action="store_true",
                        help="By default the networks will be converted to ONNX to allow TensorRT inference."
                             " If this parameter is enabled no conversion will be done")
    parser.add_argument('--in-memory-dataset',default=False, action="store_true",help="Keep single dataset in memory")
    parser.add_argument('--use_wandb', type=parse_bool, default=True, help='whether use "weights & biases" for tracking metrics, video recordings and model checkpoints')
    parser.add_argument('--continue-runs', type=parse_bool, default=True, help='continue last run?')

    args = parser.parse_args(cmd_args)

    if args.context not in ["cpu", "gpu"]:
        raise ValueError('Given value: %s for context is invalid. It must be in ["cpu", "gpu"].' % args.context)

    return args


def main():
    """
    Main function which is executed on start-up. If you train on multiple GPUs, start the
    trainer GPU before the generating GPUs to get correct epoch counting and process/binary naming.
    :return:
    """
    args = parse_args(sys.argv[1:])
    rl_config = RLConfig()

    if not os.path.exists(rl_config.binary_dir):
        raise Exception(f'Your given binary_dir: {os.path.abspath(os.path.realpath(rl_config.binary_dir))} does not exist. '
                        f'Make sure to define a valid directory')
    if rl_config.binary_dir[-1] != '/':
        rl_config.binary_dir += '/'

    enable_logging(logging.INFO, get_log_filename(args, rl_config))

    rl_loop = RLLoop(args, rl_config, nb_arena_games=rl_config.arena_games, lr_reduction=0)
    if args.trainer:
        rl_loop.current_binary_name = change_binary_name(rl_loop.file_io.binary_dir, rl_loop.current_binary_name,
                                                         rl_loop.rtpt._get_title(), rl_loop.nn_update_index)
    else:
        rl_loop.initialize()

    logging.info(f'--------------- CONFIG SETTINGS ---------------')
    for key, value in sorted(vars(args).items()):
        logging.info(f'CMD line args:      {key} = {value}')
    for key, value in sorted(dataclasses.asdict(rl_loop.tc).items()):
        logging.info(f'Train Config:       {key} = {value}')
    for key, value in sorted(dataclasses.asdict(rl_config).items()):
        logging.info(f'RL Options:         {key} = {value}')
    if not args.trainer:
        for key, value in rl_loop.binary_io.get_uci_options().items():
            logging.info(f'UCI Options:        {key} = {value}')
        for key, value in sorted(dataclasses.asdict(UCIConfigArena()).items()):
            logging.info(f'UCI Options Arena:  {key} = {value}')
    logging.info(f'-----------------------------------------------')

    while True:
        if args.trainer:
            rl_loop.check_for_enough_train_data(rl_config.nn_update_files)
        else:
            rl_loop.check_for_new_model()
            if args.evaluater:
                if time.time()>=rl_loop.next_winrate_eval:
                    rl_loop.evaluate()
                    rl_loop.check_for_new_model()
            
            success, statistics = rl_loop.binary_io.generate_games(rl_config.nb_selfplay_games_per_thread)

            if success:
                rl_loop.file_io.move_data_to_export_dir(rl_loop.device_name)
                statistics["stats/Binary_failed"] = 0
                wandb.log(statistics)
            else:
                wandb.log({"stats/Binary_failed":1})
                print("BINARY Errored, doing restart")
                rl_loop.check_for_new_model()
    wandb.finish()

if __name__ == "__main__":
    main()

