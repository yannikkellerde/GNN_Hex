from graph_game.hex_board_game import Hex_board, build_hex_grid
import os
from subprocess import PIPE, Popen
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys

def set_uci_param(proc, name: str, value):
    if type(value) == bool:
        value = f'true' if value else f'false'
    proc.stdin.write(b"setoption name %b value %b\n" % (bytes(name, encoding="utf-8"),
                                                             bytes(str(value), encoding="utf-8")))
    proc.stdin.flush()

def read_output(proc, last_line=b"readyok\n", check_error=True):
    look_for = ["board_moves_blue:","board_moves_red:","Evaluation:","Value:","Engine_move:","Dead_move","Engine_move:","Response:","Winner:"]
    info = dict()
    print_all = False
    while True:
        line = proc.stdout.readline()
        strline = line.strip().decode().strip()
        if check_error and line == b'':
            error = proc.stderr.readline()
            if error != b'':
                print(error)
            if print_all and error!=b"":
                print(error)
        if line!=b'':
            splitparts = strline.split(" ")
            if splitparts[0] in look_for:
                info[splitparts[0]] = splitparts[1:]
            if line == last_line:
                return info
            elif "received signal" in strline:
                print_all=True
            if print_all and line!=b"":
                print(str(line))

def play_vs_binary(binary_path, model_path):
    global hex_size,border_swap,prev_red_squares,prev_blue_squares
    border_swap = False
    prev_blue_squares = []
    prev_red_squares = []
    def onclick(event):
        plt.title("")
        if not hasattr(event,"xdata") or event.xdata is None:
            return
        click_coord = np.array([event.xdata, event.ydata])
        distances = np.sum((coords-click_coord)**2,axis=1)
        to_place = np.argmin(distances)
        print(f"sending move {to_place}")
        proc.stdin.write(f"{to_place}\n".encode())
        proc.stdin.flush()
        read_and_draw()

    def onpress(event):
        global hex_size,border_swap,prev_blue_squares,prev_red_squares
        if event.key == "r":
            prev_blue_squares = []
            prev_red_squares = []
            border_swap = False
            proc.stdin.write(b"reset\n")
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "s":
            proc.stdin.write(b"switch\n")
            print("sending switch")
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "+":
            border_swap = False
            prev_blue_squares = []
            prev_red_squares = []
            hex_size+=1
            set_coords()
            proc.stdin.write(f"reset {hex_size}\n".encode())
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "-":
            border_swap = False
            prev_blue_squares = []
            prev_red_squares = []
            hex_size-=1
            set_coords()
            proc.stdin.write(f"reset {hex_size}\n".encode())
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "e":
            proc.stdin.write(b"engine_move\n")
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "a":
            proc.stdin.write(b"show\n")
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "m":
            proc.stdin.write(b"mcts\n")
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "n":
            proc.stdin.write(b"raw\n")
            proc.stdin.flush()
            read_and_draw()
        elif event.key == "z":
            proc.stdin.write(b"swap\n")
            proc.stdin.flush()
            read_and_draw()



    def read_and_draw():
        global prev_red_squares,prev_blue_squares,border_swap
        info = read_output(proc,b"readyok\n", check_error=True)
        board = Hex_board()
        board.size = hex_size
        board.position = ["f"]*(hex_size*hex_size)
        if (info["board_moves_blue:"] == prev_red_squares and prev_red_squares) or (info["board_moves_red:"] == prev_blue_squares and prev_blue_squares):
            border_swap = not border_swap
        prev_red_squares = info["board_moves_red:"]
        prev_blue_squares = info["board_moves_blue:"]
        for move in info["board_moves_blue:"]:
            board.position[int(move)] = "b"
        for move in info["board_moves_red:"]:
            board.position[int(move)] = "r"
        policy = [""]*(hex_size*hex_size)
        if "Evaluation:" in info:
            for eval_comb in info["Evaluation:"]:
                print(eval_comb)
                board_ind, ev = eval_comb.replace("(","").replace(")","").split(",")
                if board_ind == "swap":
                    print("swap prob:",str(round(float(ev),3)))
                else:
                    policy[int(board_ind)] = str(round(float(ev),3))
        for key in just_prints:
            if key in info:
                print(f"{key} {' '.join(info[key])}")

        colors = [[("w" if y=="f" else y) for y in board.position[x:x+board.size]] for x in range(0,board.size*board.size,board.size)]
        labels = np.array(policy).reshape((board.size,board.size))
        fig.clear()
        fig.add_subplot()
        build_hex_grid(colors,labels,fig=fig,border_swap=border_swap)
    def set_coords():
        global coords
        xstart = -(hex_size//2)*1.5
        ystart = -(hex_size/2*np.sqrt(3/4))+0.5
        coords = []
        for i in range(hex_size):
            for j in range(hex_size):
                coords.append([xstart+0.5*j+i,ystart+np.sqrt(3/4)*j])
        coords = np.array(coords)
    just_prints =  ["Value:","Dead_move","Engine_move:","Response:","Winner:"]
    plt.rcParams["keymap.yscale"].remove('l')
    plt.rcParams['keymap.save'].remove('s')
    hex_size = 7
    set_coords()
    fig = plt.gcf()
    proc = Popen(["gdb","-batch","-ex",'run',"-ex",'bt',binary_path], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False)
    set_uci_param(proc,f'Model_Path', model_path)
    proc.stdin.write(b"isready\n")
    proc.stdin.flush()
    read_output(proc,b"readyok\n", check_error=True)
    proc.stdin.write(b"play\n")
    proc.stdin.flush()
    while 1:
        read_and_draw()
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onpress)
        plt.show()

if __name__ == "__main__":
    play_vs_binary(sys.argv[1],sys.argv[2])
