"""
@file: fileio.py
Created on 01.04.2021
@project: CrazyAra
@author: queensgambit, maxalexger

Contains the main class to handle files and directories during Reinforcement Learning.
Additionally a function to compress zarr datasets is provided.
"""
import os
import glob
import time
import logging
import datetime
import numpy as np
from typing import Tuple

from rl_loop.main_config import main_config
from rl_loop.train_config import TrainConfig
from rl_loop.rl_utils import create_dir, move_all_files, move_oldest_files


class FileIO:
    """
    Class to facilitate creation of directories, reading of file
    names and moving of files during Reinforcement Learning.
    """
    def __init__(self, orig_binary_name: str, binary_dir: str, model_name:str, uci_variant: str, framework: str):
        """
        Creates all necessary directories and sets all path variables.
        If no '*.param' file can be found in the 'binary-dir/model/' directory,
        we assume that every folder has another subdirectory named after the UCI-Variant.
        """
        self.binary_dir = binary_dir
        self.uci_variant = uci_variant
        self.framework = framework
        self.model_name = model_name

        # If there is no model in 'model/', we assume that the model and every
        # other path has an additional '<variant>' folder
        variant_suffix = f''

        # Hard coded directory paths
        self.binary_data_output = os.path.join(binary_dir,"data")
        self.model_dir = os.path.join(binary_dir, "model", orig_binary_name)
        self.export_dir_gen_data = os.path.join(binary_dir, "export","new_data")
        self.train_dir = os.path.join(binary_dir, "export/train")
        self.val_dir = os.path.join(binary_dir, "export/val")
        self.weight_dir = os.path.join(binary_dir, "weights")
        self.train_dir_archive = os.path.join(binary_dir, "export/archive/train")
        self.val_dir_archive = os.path.join(binary_dir, "export/archive/val")
        self.model_contender_dir = os.path.join(binary_dir, "model_contender", orig_binary_name)
        self.model_dir_archive = os.path.join(binary_dir, "export/archive/model")
        self.logs_dir_archive = os.path.join(binary_dir, "export/logs")
        self.logs_dir = os.path.join(binary_dir, "logs")
        self.arena_pgns_dir = os.path.join(binary_dir, "arena_pgns")

        self._create_directories()

        # Adjust paths in main_config
        main_config["planes_train_dir"] = os.path.join(binary_dir, "export/train/")
        main_config["planes_val_dir"] = os.path.join(binary_dir, "export/val/")
        assert os.path.isdir(main_config["planes_train_dir"]) is not False, \
            f'Please provide valid main_config["planes_train_dir"] directory'

    def store_arena_pgn(self,model_num):
        os.rename(os.path.join(self.binary_data_output,"arena_games.pgn"),os.path.join(self.arena_pgns_dir,f"arena_games_{model_num}.pgn"))

    def _create_directories(self):
        """
        Creates directories in the binary folder which will be used during RL
        :return:
        """
        create_dir(self.logs_dir)
        create_dir(self.weight_dir)
        create_dir(self.export_dir_gen_data)
        create_dir(self.train_dir)
        create_dir(self.val_dir)
        create_dir(self.train_dir_archive)
        create_dir(self.val_dir_archive)
        create_dir(self.model_contender_dir)
        create_dir(self.model_dir_archive)
        create_dir(self.model_dir)
        create_dir(self.logs_dir_archive)
        create_dir(self.arena_pgns_dir)

    def _include_data_from_replay_memory(self, nb_files: int, fraction_for_selection: float):
        """
        :param nb_files: Number of files to include from replay memory into training
        :param fraction_for_selection: Proportion for selecting files from the replay memory
        :return:
        """
        file_names = os.listdir(self.train_dir_archive)

        file_names.sort(key=lambda x:os.path.getmtime(os.path.join(self.train_dir_archive,x)))
        # invert ordering (most recent files are on top)
        file_names.reverse()

        if len(file_names) < nb_files:
            logging.info("Not enough replay memory available. Only current data will be used")
            return

        thresh_idx = max(int(len(file_names) * fraction_for_selection), nb_files)

        indices = np.arange(0, thresh_idx)
        np.random.shuffle(indices)

        # cap the index list
        indices = indices[:nb_files]

        # move selected files into train dir
        for index in list(indices):
            os.rename(os.path.join(self.train_dir_archive, file_names[index]), os.path.join(self.train_dir, file_names[index]))

    def _move_generated_data_to_train_val(self):
        """
        Moves the generated samples, games (pgn format) and the number how many games have been generated to the given
        training and validation directory
        :return:
        """
        file_names = os.listdir(self.export_dir_gen_data)

        # move the last file into the validation directory
        os.rename(os.path.join(self.export_dir_gen_data, file_names[-1]), os.path.join(self.val_dir, file_names[-1]))

        # move the rest into the training directory
        for file_name in file_names[:-1]:
            os.rename(os.path.join(self.export_dir_gen_data, file_name), os.path.join(self.train_dir, file_name))

    def _move_train_val_data_into_archive(self):
        """
        Moves files from training, validation dir into archive directory
        :return:
        """
        move_oldest_files(self.train_dir, self.train_dir_archive, TrainConfig.training_keep_files)
        move_all_files(self.val_dir, self.val_dir_archive)

    def _remove_files_in_weight_dir(self):
        """
        Removes all files in the weight directory.
        :return:
        """
        file_list = glob.glob(os.path.join(self.weight_dir, "model-*"))
        for file in file_list:
            os.remove(file)

    def move_data_to_export_dir(self, device_name: str):
        """
        Loads the uncompressed data file, selects all sample until the index specified in "startIdx.txt",
        compresses it and exports it.
        :param device_name: The currently active device name (context_device-id)
        :return:
        """
        data_folders = [os.path.join(self.binary_data_output,x) for x in os.listdir(self.binary_data_output) if os.path.isdir(os.path.join(self.binary_data_output,x))]
        for folder in data_folders:
            export_dir, time_stamp = self.create_export_dir(device_name,os.path.basename(folder))
            os.rename(os.path.join(folder,"torch"),os.path.join(export_dir,"torch"))
            os.rename(os.path.join(folder, "games.pgn"), os.path.join(export_dir, "games.pgn"))

    def create_export_dir(self, device_name: str, idx) -> Tuple[str, str]:
        """
        Create a directory in the 'export_dir_gen_data' path,
        where the name consists of the current date, time and device ID.
        :param device_name: The currently active device name (context_device-id)
        :return: Path of the created directory; Time stamp used while creating
        """
        # include current timestamp in dataset export file
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        time_stamp_dir = os.path.join(self.export_dir_gen_data,f'{time_stamp}-{idx}-{device_name}/')
        # create a directory of the current time_stamp
        if not os.path.exists(time_stamp_dir):
            os.makedirs(time_stamp_dir)

        return time_stamp_dir, time_stamp

    def get_current_model_pt_file(self) -> str:
        """
        Return the filename of the current active model weight (.pt) file for pytorch
        """
        for fname in os.listdir(self.model_dir):
            if fname.startswith("weights"):
                return os.path.join(self.model_dir,fname)
        raise FileNotFoundError("Model weights file not found")

    def get_number_generated_files(self) -> int:
        """
        Returns the amount of file that have been generated since the last training run.
        :return:
        """
        return len([x for x in os.listdir(self.export_dir_gen_data) if os.path.isdir(os.path.join(self.export_dir_gen_data,x))])

    def move_training_logs(self, nn_update_index):
        """
        Rename logs and move it from /logs to /export/logs/
        """
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        dir_name = f'logs-{self.uci_variant}-update{nn_update_index}-{time_stamp}'
        os.rename(self.logs_dir, os.path.join(self.logs_dir_archive, dir_name))
        create_dir(self.logs_dir)

    def prepare_data_for_training(self, rm_nb_files: int, rm_fraction_for_selection: float, did_contender_win: bool):
        """
        Move files from training, validation and model contender folder into archive.
        Moves newly generated files into training and validation directory.
        Remove files in weight directory. Include data from replay memory.
        :param rm_nb_files: Number of files of the replay memory to include
        :param rm_fraction_for_selection: Proportion for selecting files from the replay memory
        :param did_contender_win: Defines if the last contender won vs the generator
        """
        if did_contender_win:
            self._move_train_val_data_into_archive()
        # move last contender into archive
        move_all_files(self.model_contender_dir, self.model_dir_archive)

        self._move_generated_data_to_train_val()
        # We don’t need them anymore; the last model from last training has already been saved
        self._remove_files_in_weight_dir()
        self._include_data_from_replay_memory(rm_nb_files, rm_fraction_for_selection)

    def remove_intermediate_weight_files(self):
        """
        Deletes all files (excluding folders) inside the weight directory
        """
        # Replace _weight_dir with self.weight_dir, if the trainer can save weights dynamically
        _weight_dir = os.path.join(self.binary_dir, 'weights')
        files = glob.glob(_weight_dir + '/model-*')
        for f in files:
            os.remove(f)

    def replace_current_model_with_contender(self):
        """
        Moves the previous model into archive directory and the model-contender into the model directory
        """
        move_all_files(self.model_dir, self.model_dir_archive)
        move_all_files(self.model_contender_dir, self.model_dir)
