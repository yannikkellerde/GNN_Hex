import matplotlib.pyplot as plt
from graph_game.hex_board_game import build_hex_grid
import numpy as np
import sys

def show_eval_from_file(fname:str,colored=".5",fontsize=8):
    with open(fname,"r") as f:
        evals = [float(x) for x in f.readline().split(" ") if len(x)>0]
    if colored == "sign":
        colors = ["r" if e>0 else "b" for e in evals]
    elif colored == ".5":
        colors = ["r" if e>0.5 else "b" for e in evals]
    elif colored == "none":
        colors = ["w" for _ in evals]
    elif colored == "top3":
        colors = ["w" for _ in evals]
        sort_idx = np.argsort(evals)
        colors[sort_idx[-1]] = "darkred"
        colors[sort_idx[-2]] = "darkred"
        colors[sort_idx[-3]] = "red"
        colors[sort_idx[-4]] = "red"
        colors[sort_idx[-5]] = "orange"
        colors[sort_idx[-6]] = "orange"
    else:
        raise ValueError("Invalid argument")

    hex_size = np.sqrt(len(evals))
    assert int(hex_size) == hex_size
    hex_size = int(hex_size)

    colors = np.array(colors).reshape((hex_size,hex_size))
    labels = [f"{e:.3f}" for e in evals]
    labels = np.array(labels).reshape((hex_size,hex_size))

    fig = plt.gcf()
    fig.clear()
    fig.add_subplot()
    return build_hex_grid(colors,labels,fig=fig,do_pause=False,fontsize=fontsize)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        show_eval_from_file(sys.argv[1])
    elif len(sys.argv) == 3:
        show_eval_from_file(sys.argv[1],sys.argv[2])
    else:
        raise ValueError("Invalid amount of command line args")
    plt.show()
