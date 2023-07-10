from subprocess import Popen, PIPE, STDOUT
from threading  import Thread
from queue import Queue, Empty
import time
from graph_game.hex_board_game import Hex_board
from graph_game.graph_tools_games import Hex_game
from typing import List
import os


class MohexSelfplay():
    def __init__(self,binary_path="./mohex",max_games=99999999,max_time=10,num_parallel_games=1,hex_size=7,tmp_dump="tmp"):
        self.hex_size = hex_size
        self.tmp_dump = tmp_dump
        os.makedirs(self.tmp_dump,exist_ok=True)
        self.procs = []
        self.hex_games = []
        self.move_information = []
        for i in range(num_parallel_games):
            proc = Popen(["gdb","-batch","-ex",'run',"-ex",'bt',binary_path], stdin=PIPE, stdout=PIPE, stderr=STDOUT, shell=False)
            os.set_blocking(proc.stdout.fileno(), False)
            proc.stdin.write(f"param_mohex max_time {max_time}\n".encode())
            proc.stdin.flush()
            self._wait_for_answer(proc,"=")
            proc.stdin.write(f"param_mohex max_games {max_games}\n".encode())
            proc.stdin.flush()
            self._wait_for_answer(proc,"=")
            self.procs.append(proc)
            game = Hex_game(hex_size)
            self.hex_games.append(game)

    def play_n_games(self,num_games,random_first_n_moves=0,safety_writes_every=None):
        games_left = num_games-len(self.hex_games)
        dones = [False]*len(self.hex_games)
        self.hex_games = [Hex_game(self.hex_size) for _ in self.procs]
        for game in self.hex_games:
            game.board_callback = game.board.graph_callback
        move_information = [[[]] for _ in self.procs]
        
        while not all(dones):
            print("start make moves")
            for i,(game,mi) in enumerate(zip(self.hex_games,move_information)):
                if dones[i]:
                    continue
                winner = game.who_won()
                if winner is not None:
                    if winner == "m":
                        value = 1
                    else:
                        value = -1
                    my_list = mi[-1]
                    for di in my_list:
                        di["value"] = value
                        value = -value
                    if games_left>0:
                        self.hex_games[i] = Hex_game(self.hex_size)
                        self.hex_games[i].board_callback = self.hex_games[i].board.graph_callback
                        move_information[i].append([])
                        games_left-=1
                        if safety_writes_every is not None and (num_games-len(self.hex_games)-games_left)%safety_writes_every==0:
                            self.dump_move_info_to_file(move_information)

                    else:
                        dones[i] = True

            valid_i_game_proc = [(i,game,proc) for i,(game,proc,done) in enumerate(zip(self.hex_games,self.procs,dones)) if not done]
            for i,game,proc in valid_i_game_proc:
                with open(f"{self.tmp_dump}/cur_game_state_{i}.sgf","w") as f:
                    f.write(game.board.to_sgf())
                proc.stdin.write(f"loadsgf {self.tmp_dump}/cur_game_state_{i}.sgf\n".encode())
                proc.stdin.flush()
            print("wait for sgf answers")
            self.wait_for_all_answers([x[2] for x in valid_i_game_proc],"=")
            print("loaded sgfs")

            for i,game,proc in valid_i_game_proc:
                color = "w" if game.view.gp["m"] else "b"
                proc.stdin.write(f"genmove {color}\n".encode())
                proc.stdin.flush()
            checks,lines = self.wait_for_all_answers([x[2] for x in valid_i_game_proc],"=",check_for="Score")
            for j in range(len(lines)):
                i = valid_i_game_proc[j][0]
                info_dict = {}
                info_list = move_information[i][-1]
                line = lines[j]
                move_notation = line.split(" ")[1].replace("\\n","").replace("'","")
                move = valid_i_game_proc[j][1].board.notation_to_number(move_notation)
                info_dict["best_move"] = move
                game = valid_i_game_proc[j][1]
                if len(info_list)>=random_first_n_moves:
                    info_dict["played_move"] = move
                else:
                    info_dict["played_move"] = game.board.sample_legal_move()
                    
                game.make_move(game.board.board_index_to_vertex[info_dict["played_move"]],remove_dead_and_captured=True)
                # print(game.board.draw_me())
                info_dict["swap_prob"] = 0
                if len(info_list)==1:
                    score = float(checks[j].replace("Score","").strip().replace("\\n",""))
                    info_dict["swap_prob"] = 1-score
                info_list.append(info_dict)
        self.dump_move_info_to_file(move_information)

    def dump_move_info_to_file(self,move_info):
        move_info = sum(move_info,[])
        with open("mohex_data.txt","w") as f:
            for game_data in move_info:
                f.write("New game\n")
                for move_data in game_data:
                    if 'value' in move_data:
                        f.write(f"{move_data['played_move']},{move_data['best_move']},{move_data['swap_prob']},{move_data['value']}\n")
        return move_info

    def wait_for_all_answers(self,procs=None,start_to_look_for="=",check_for=None):   
        if procs is None:
            procs = self.procs
        lines = [None]*len(procs)
        checks = [None]*len(procs)
        print("######  New wait for answers  ###########")
        while any([x is None for x in lines]):
            for i,proc in enumerate(procs):
                line = proc.stdout.readline()
                line = line.decode("utf-8")  # bytes to string
                # if len(line)>0:
                #     print(f"proc {i}",line)
                line=line.strip()
                # else:
                #     while error:=self.proc.stderr.readline() != b'':
                #         print("Error",error)
                if line.startswith(start_to_look_for):
                    lines[i] = line
                if check_for is not None and line.startswith(check_for):
                    checks[i] = line

        return checks,lines

    def _wait_for_answer(self,proc,start_to_look_for="="):
        infos_found = {}
        while True:
            line = proc.stdout.readline()
            print(line)
            line = line.decode("utf-8")  # bytes to string
            # if len(line)>0:
            #     print(line)
            line=line.strip()
            # else:
            #     while error:=self.proc.stderr.readline() != b'':
            #         print("Error",error)
            if line.startswith(start_to_look_for):
                return line

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

if __name__ == "__main__":
    gen = MohexSelfplay(binary_path="mohex",hex_size=11,max_games=1000,num_parallel_games=1)
    gen.play_n_games(10000,random_first_n_moves=3,safety_writes_every=500)
