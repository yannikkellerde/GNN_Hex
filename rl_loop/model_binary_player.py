from subprocess import Popen, PIPE, STDOUT
import time
from graph_game.hex_board_game import Hex_board
from graph_game.graph_tools_games import Hex_game
from typing import List
import os

class BinaryPlayer():
    def __init__(self,model_path,binary_path,use_mcts,hex_size=11):
        self.proc = Popen(["gdb","-batch","-ex",'run',"-ex",'bt',binary_path], stdin=PIPE, stdout=PIPE, stderr=STDOUT, shell=False)
        self.proc.stdin.write(f"setoption name Model_Path value {model_path}\n".encode())
        self.proc.stdin.flush()
        self.proc.stdin.write(f"setoption name Hex_Size value {hex_size}\n".encode())
        self.proc.stdin.flush()
        self.proc.stdin.write(f"isready\n".encode())
        self.proc.stdin.flush()
        self._wait_for_answer("readyok")
        print("loaded model")
        self.proc.stdin.write(f"play\n".encode())
        self.proc.stdin.flush()
        self._wait_for_answer("readyok")
        print("In play mode")
        self.proc.stdin.write(b"set_autoeval false\n")
        self.proc.stdin.flush()
        print("autoeval set")
        self._wait_for_answer("readyok")
        self.proc.stdin.write(b"mcts\n" if use_mcts else b"raw\n")
        self.proc.stdin.flush()
        self._wait_for_answer("readyok")
        print("Initialized HexAra")

    def __call__(self,games:List[Hex_game]):
        moves = []
        move_notations = []
        for i,game in enumerate(games):
            print("loading sgf")
            print("Did a player win?",game.who_won())
            color = "w" if game.view.gp["m"] else "b"
            curpath = os.path.abspath(os.getcwd())
            with open(os.path.join(curpath,f"cur_game_state.sgf"),"w") as f:
                f.write(game.board.to_sgf())
            self.proc.stdin.write(f"sgf {os.path.join(curpath,f'cur_game_state.sgf')}\n".encode())
            self.proc.stdin.flush()
            self._wait_for_answer("readyok")
            self.proc.stdin.write(b"eval\n")
            self.proc.stdin.flush()
            self._wait_for_answer("readyok")
            self.proc.stdin.write(b"engine_move\n")
            self.proc.stdin.flush()
            info,line = self._wait_for_answer("readyok",stats_to_log=["Engine_move:"])
            move = int(info["Engine_move:"].split(" ")[1].replace("\\n",""))
            moves.append(move)
        print(moves)
        return moves

    def _wait_for_answer(self,start_to_look_for="readyok",stats_to_log=[]):
        infos_found = {}
        while True:
            line = self.proc.stdout.readline()
            line = line.decode("utf-8")  # bytes to string
            if len(line)>0:
                print(line)
            line=line.strip()
            # else:
            #     while error:=self.proc.stderr.readline() != b'':
            #         print("Error",error)
            if line.startswith(start_to_look_for):
                return infos_found,line
            for x in stats_to_log:
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
