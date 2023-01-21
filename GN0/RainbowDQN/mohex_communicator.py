from subprocess import Popen, PIPE, STDOUT
import time
from graph_game.hex_board_game import Hex_board
from graph_game.graph_tools_games import Hex_game
from typing import List
import os

class MohexPlayer():
    def __init__(self,binary_path="./mohex",max_games=99999999,max_time=10):
        self.proc = Popen(["gdb","-batch","-ex",'run',"-ex",'bt',binary_path], stdin=PIPE, stdout=PIPE, stderr=STDOUT, shell=False)
        self.proc.stdin.write(f"param_mohex max_time {max_time}\n".encode())
        self.proc.stdin.flush()
        self._wait_for_answer("=")
        self.proc.stdin.write(f"param_mohex max_games {max_games}\n".encode())
        self.proc.stdin.flush()
        self._wait_for_answer("=")

    def __call__(self,games:List[Hex_game]):
        moves = []
        move_notations = []
        for game in games:
            color = "w" if game.view.gp["m"] else "b"
            with open("cur_game_state.sgf","w") as f:
                f.write(game.board.to_sgf())
            self.proc.stdin.write(b"loadsgf cur_game_state.sgf\n")
            self.proc.stdin.flush()
            self._wait_for_answer("=")
            self.proc.stdin.write(f"genmove {color}\n".encode())
            self.proc.stdin.flush()
            infos,line = self._wait_for_answer("=")
            move_notation = line.split(" ")[1].replace("\\n","").replace("'","")
            move = game.board.notation_to_number(move_notation)
            moves.append(move)
            move_notations.append(move_notation)
        print(moves)
        print(move_notations)
        return moves

    def _wait_for_answer(self,start_to_look_for="=",starts_to_log=[]):
        infos_found = {}
        while True:
            line = self.proc.stdout.readline()
            line = line.decode("utf-8").strip()  # bytes to string
            if len(line)>0:
                print(line)
            # else:
            #     while error:=self.proc.stderr.readline() != b'':
            #         print("Error",error)
            if line.startswith(start_to_look_for):
                return infos_found,line
            for x in start_to_look_for:
                if line.startswith(x):
                    infos_found[x] = line


    def stop_process(self,fail_ok=False):
        """
        Kills the process that is attached to the binary.
        :return:
        """
        if fail_ok:
            try:
                self.proc.kill()
            except Exception as e:
                print("Process kill",e)
        else:
            self.proc.kill()
        # sleep for 1 sec to ensure the process exited
        time.sleep(1)

