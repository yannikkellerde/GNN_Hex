import matplotlib.pyplot as plt
from rl_loop.train_util import input_planes_from_position,get_starting_position
import torch
from matplotlib.patches import RegularPolygon
import numpy as np
import sys

def build_hex_grid(colors,labels=None,fig=None,border_swap=False,do_pause=True,fontsize=14):
    if labels is not None:
        labels = [[str(x)[:6] for x in y] for y in labels]
    if fig is not None:
        ax = fig.axes[0]
    else:
        fig, ax = plt.subplots(1,figsize=(16,16))
    size = len(colors)
    xstart = -(size//2)*1.5
    ystart = -(size/2*np.sqrt(3/4))+0.5
    xend = xstart+1.5*(size-1)
    yend = ystart+np.sqrt(3/4)*(size-1)
    ax.set_aspect('equal')
    tri = plt.Polygon([[0,0],[xstart-1.25,ystart-0.75],[xstart+0.5*(size-1)-0.5,yend+0.75]],color="b" if border_swap else "r",alpha=0.7)
    ax.add_patch(tri)
    tri = plt.Polygon([[0,0],[xend+1.25,yend+0.75],[xstart+0.5*(size-1)-0.5,yend+0.75]],color="r" if border_swap else "b",alpha=0.7)
    ax.add_patch(tri)
    tri = plt.Polygon([[0,0],[xend+1.25,yend+0.75],[xstart+1*(size-1)+0.5,ystart-0.75]],color="b" if border_swap else "r",alpha=0.7)
    ax.add_patch(tri)
    tri = plt.Polygon([[0,0],[xstart-1.25,ystart-0.75],[xstart+1*(size-1)+0.5,ystart-0.75]],color="r" if border_swap else "b",alpha=0.7)
    ax.add_patch(tri)
    for i,cylist in enumerate(colors):
        for j,color in enumerate(cylist):
            coords = [xstart+0.5*j+i,ystart+np.sqrt(3/4)*j]
            hexagon = RegularPolygon((coords[0], coords[1]), numVertices=6, radius=np.sqrt(1/3), alpha=1, edgecolor='k', facecolor=color,linewidth=2)
            if labels is not None:
                ax.text(coords[0]-0.4, coords[1]-0.05,labels[i][j],color="black" if (color=="white" or color=="w") else "white",fontsize=fontsize)
            ax.add_patch(hexagon)
    plt.autoscale(enable=True)
    plt.axis("off")
    plt.tight_layout()
    if do_pause:
        plt.pause(0.001)
    return fig

def compute_and_plot_starting_eval(hex_size,model,color="r",fontsize=8,device="cpu"):
    input = input_planes_from_position(get_starting_position(hex_size),color)
    policy,value = model(input.unsqueeze(0).to(device))
    policy = policy.squeeze()
    colors = ["w" for _ in policy]
    sort_idx = torch.argsort(policy)
    colors[int(sort_idx[-1])] = "darkred"
    colors[int(sort_idx[-2])] = "darkred"
    colors[int(sort_idx[-3])] = "red"
    colors[int(sort_idx[-4])] = "red"
    colors[int(sort_idx[-5])] = "orange"
    colors[int(sort_idx[-6])] = "orange"
    colors = np.array(colors).reshape((hex_size,hex_size))
    labels = [f"{e:.3f}" for e in policy]
    labels = np.array(labels).reshape((hex_size,hex_size))
    fig = plt.gcf()
    fig.clear()
    fig.add_subplot()
    return build_hex_grid(colors,labels,fig=fig,do_pause=False,fontsize=fontsize)

def compute_and_plot_swapmap(hex_size,model,starting_player="r",fontsize=8,device="cpu"):
    next_color = "b" if starting_player=="r" else "b"
    start_pos = get_starting_position(hex_size)
    values = []
    for i in range(len(start_pos)):
        position = start_pos.copy()
        position[i] = starting_player
        input = input_planes_from_position(position,next_color)
        policy,value = model(input.unsqueeze(0).to(device))
        value = value.squeeze()
        values.append(value.cpu().item())
    swap_probs = 1-((np.array(values)+1)/2)
    colors = ["r" if e>0.5 else "b" for e in swap_probs]
    colors = np.array(colors).reshape((hex_size,hex_size))

    labels = [f"{e:.3f}" for e in swap_probs]
    labels = np.array(labels).reshape((hex_size,hex_size))

    fig = plt.gcf()
    fig.clear()
    fig.add_subplot()
    return build_hex_grid(colors,labels,fig=fig,do_pause=False,fontsize=fontsize)

    
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
