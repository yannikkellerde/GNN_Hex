"""
@file: binaryio.py
Created on 01.04.2021
@project: CrazyAra
@author: queensgambit, maxalexger

Contains the main class to communicate with the C++ binary.
"""
import logging
import time
import os
from typing import Tuple

from subprocess import PIPE, Popen
from dataclasses import fields
from rl_loop.rl_config import UCIConfig, UCIConfigArena
from rl_loop.rl_utils import log_to_file_and_print


class BinaryIO:
    """
    This class establishes a connection to the binary and handles
    the binary inputs and outputs.
    """
    def __init__(self, binary_path: str):
        """
        Open a process to the binary in order to send commands and read output
        :param binary_path: Path to the binary including the binary name
        """
        self.binary_dir = os.path.dirname(binary_path)
        self.proc = Popen(["gdb","-batch","-ex",'run',"-ex",'bt',binary_path], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False)
        # self.proc = Popen([binary_path], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False)

    def generate_starting_eval_img(self):
        self.proc.stdin.write(b"starting_eval\n")
        self.proc.stdin.flush()
        self.read_output(b"readyok\n", check_error=True)
        self.proc.stdin.write(b"swapmap\n")
        self.proc.stdin.flush()
        self.read_output(b"readyok\n", check_error=True)


    def compare_new_weights(self, nb_arena_games: int, threads:int) -> Tuple[bool,float]:
        """
        Compares the old NN-weights with the newly acquired one.
        Internally, the binary uses the UCI Options 'Model_Directory' &
        'Model_Directory_contender' to read in the old and new NN-weights respectively.
        :param nb_arena_games: Number of non-standoff games btw old and new NN-weights
        :return: True - If current NN generator should be replaced
                 False - If current NN generator should be kept
        """
        arena_games_per_thread = (nb_arena_games//threads)
        if arena_games_per_thread%2==1:
            arena_games_per_thread+=1
        self.proc.stdin.write(f"arena {nb_arena_games} {threads}\n".encode())
        self.proc.stdin.flush()
        return self.read_output_arena(check_error=True)

    def generate_games(self, nb_games_per_thread):
        """
        Requests the binary to generate games. The number of games generated by the
        binary is determined by Selfplay_Number_Chunks * Selfplay_Chunk_Size.
        Both variables are UCI options within the binary.
        The ouput of the binary will be inside the binarz folder and contain a 'data.zarr'
        folder with the compressed games, as well as 'games.pgn' and 'gameIdx.txt'.
        :return:
        """
        logging.info(f'Generating games ...')
        # if 0, binary plays Selfplay_Number_Chunks * Selfplay_Chunk_Size games
        self.proc.stdin.write(f"selfplay {nb_games_per_thread}\n".encode())
        self.proc.stdin.flush()
        return self.read_output(b"readyok\n", check_error=True)

    def get_uci_options(self):
        """
        Requests and reads uci options from a process.
        This is non UCI standard behavior! After sending 'activeuci' the
        process returns all the current options in the format
        'option name <name> value <value>' followed by 'readyok' at the end.

        :return: Dictionary with the option's names as keys and option's values as values
        """
        options = {}
        self.proc.stdin.write(b'activeuci\n')  # write bytes
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            line = line.decode("utf-8").strip()  # bytes to string
            if line == f'readyok':
                break
            elif line.startswith(f'option name ') and f' value' in line:
                idx = line.index(f' value')
                value = line[idx+7:]
                if value.replace('.','',1).isdigit():
                    value = float(value)
                options[line[12:idx]] = value
            elif len(line) == 0:
                continue
            else:
                raise ValueError(f'uci command activeuci returned wrong format')
        return options

    def load_network(self):
        """
        Tells the binary to load the network and waits until it's finished.
        :return:
        """
        logging.info(f'Loading network & creating backend files ...', )
        self.proc.stdin.write(b"isready\n")
        self.proc.stdin.flush()
        self.read_output(b"readyok\n", check_error=True)

    def read_output(self, last_line=b"readyok\n", check_error=True):
        """
        Reads the output of a process pip until the given last line has been reached.
        :param last_line Content when to stop reading (e.g. b'\n', b'', b"readyok\n")
        :param check_error: Listens to stdout for errors
        :return:
        """
        look_fors = ["50 games","WARNING"]
        print_all = False
        killit = False
        statistics = {}
        while True:
            line = self.proc.stdout.readline()
            strline = str(line)
            if check_error and line == b'':
                error = self.proc.stderr.readline()
                if error != b'':
                    logging.error(error)
                elif "received signal" in str(error):
                    killit = time.perf_counter()+3
                    print_all=True
                if print_all and error!=b"":
                    print(error)
            if line == last_line:
                return True, statistics
            elif any([x in strline for x in look_fors]):
                logging.debug(line)
            elif "received signal" in strline:
                killit = time.perf_counter()+3
                print_all=True
            if "Statistic:" in strline:
                parts = strline.strip().replace("\\n","").replace("'","").split(" ")
                statistics["stats/"+parts[1]] = float(parts[2])
            if print_all and line!=b"":
                log_to_file_and_print(os.path.join(self.binary_dir,"logs","errors.log"),str(line))
            elif killit and time.perf_counter()>killit:
                return False, statistics

    def read_output_arena(self, check_error=True) -> Tuple[bool,float]:
        """
        Reads the output for arena matches and waits for the key-words "keep" or "replace"
        :param check_error: Listens to stdout for errors
        :return: True - If current NN generator should be replaced
                 False - If current NN generator should be kept
        """
        winrate = -100
        print_all = False
        killit = False
        while True:
            line = self.proc.stdout.readline()
            strline = str(line)
            if check_error and line == b'':
                error = self.proc.stderr.readline()
                if error != b'':
                    logging.error(error)
            elif "received signal" in strline:
                killit = time.perf_counter()+3
                print_all=True
            elif line.startswith(b"Contender winrate:"):
                parts = strline.strip().replace("\\n","").replace("'","").split(" ")
                winrate = float(parts[2])
                logging.info(line)
            elif line == b"keep\n":
                return False,winrate
            elif line == b"replace\n":
                return True,winrate
            if print_all and line!=b"":
                log_to_file_and_print(os.path.join(self.binary_dir,"logs","errors.log"),str(line))
            elif killit and time.perf_counter()>killit:
                return False, 0

    def set_uci_options(self, uci_variant: str, context: str, device_id: str, precision: str, model_dir: str, model_contender_dir: str, threads:int, model_name:str, is_arena: bool = False):
        """
        Sets UCI options of the binary.
        :param uci_variant: The UCI variant that shall be trained.
        :param context: The context of the process (in ["cpu", "gpu"]).
        :param device_id: The id of the device we are using.
        :param precision: The precision of calculations.
        :param model_dir: The path to the model.
        :param model_contender_dir: Directory where the model contender dir will be saved.
        :param is_arena: Applies setting for the arena comparison
        :return:
        """
        self._set_uci_param(f'Model_Path', os.path.join(model_dir,model_name+"_model.pt"))
        self._set_uci_param(f'Model_Path_Contender', os.path.join(model_contender_dir,model_name+"_model.pt"))
        self._set_uci_param(f'UCI_Variant', uci_variant)
        self._set_uci_param(f'Context', context)
        self._set_uci_param(f'First_Device_ID', device_id)
        self._set_uci_param(f'Last_Device_ID', device_id)
        self._set_uci_param(f'Precision', precision)
        self._set_uci_param(f'Threads', threads)

        uci = UCIConfig()
        uci_arena = UCIConfigArena()

        # Send all UCI options (from basic UCIConfig class) that won't be sent further down
        for field in fields(uci):
            if not is_arena or (is_arena and field.name not in fields(uci_arena)):
                print(f"setting option {field.name}", getattr(uci,field.name))
                self._set_uci_param(field.name, getattr(uci, field.name))

        if is_arena:
            for field in fields(uci_arena):
                self._set_uci_param(field.name, getattr(uci_arena, field.name))

    def _set_uci_param(self, name: str, value):
        """
        Sets the value for a given UCI-parameter in the binary.
        :param name: Name of the UCI-parameter
        :param value: Value for the UCI-parameter (this value is always converted to a string)
        :return:
        """
        if type(value) == bool:
            value = f'true' if value else f'false'
        self.proc.stdin.write(b"setoption name %b value %b\n" % (bytes(name, encoding="utf-8"),
                                                                 bytes(str(value), encoding="utf-8")))
        self.proc.stdin.flush()

    def stop_process(self):
        """
        Kills the process that is attached to the binary.
        :return:
        """
        self.proc.kill()
        # sleep for 1 sec to ensure the process exited
        time.sleep(1)
