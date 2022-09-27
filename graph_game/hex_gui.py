import matplotlib.pyplot as plt
import os
from graph_game.hex_board_game import build_hex_grid
import numpy as np
from graph_game.graph_tools_games import Hex_game
from graph_game.shannon_node_switching_game import Node_switching_game
from GN0.convert_graph import convert_node_switching_game
import torch
import random
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_mover_model(game:Node_switching_game,respond_to=None):
    actions = game.get_actions()
    if len(actions) == 0:
        return random.choice(game.board.get_actions())
    else:
        return random.choice([game.board.vertex_index_to_board_index[x] for x in actions])

def random_evaluater(game,respond_to=None):
    vprop = game.view.new_vertex_property("float")
    vprop.fa = np.random.random(len(vprop.fa))
    return vprop

def playerify_model(model):
    def model_player(game:Node_switching_game,respond_to=None):
        if respond_to is not None and game.graph.vp.f[respond_to]:
            respond_to = None
        if respond_to is None or game.get_response(respond_to,game.view.gp["m"]) is None:
            if respond_to is not None:
                plt.title("Last move was dead")
            data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(device)
            res = model(data.x,data.edge_index).squeeze()
            raw_move = torch.argmax(res[2:]).item()+2
            move = data.backmap[raw_move].item()
            board_move = game.board.vertex_index_to_board_index[move]
        else:
            response = game.get_response(respond_to,game.view.gp["m"])
            plt.title("Last move was captured")
            board_move = response
        return board_move
    return model_player

def playerify_maker_breaker(maker,breaker):
    player_maker = playerify_model(maker)
    player_breaker = playerify_model(breaker)
    def maker_breaker_player(game,respond_to=None):
        if game.view.gp["m"]:
            return player_maker(game)
        else:
            return player_breaker(game)
    return maker_breaker_player

def model_to_evaluater(model):
    def evaluater(game:Node_switching_game,respond_to=None):
        if respond_to is not None and game.graph.vp.f[respond_to]:
            respond_to = None
        if respond_to is None or game.get_response(respond_to,game.view.gp["m"]) is None:
            if respond_to is not None:
                plt.title("Last move was dead")
            data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True).to(device)
            res = model(data.x,data.edge_index).squeeze()
            vinds = {data.backmap[int(i)]:value for i,value in enumerate(res) if int(i)>1}
            vprop = game.view.new_vertex_property("float")
            for key,value in vinds.items():
                vprop[game.view.vertex(key)] = value
        else:
            response = game.get_response(respond_to,game.view.gp["m"])
            vprop = game.view.new_vertex_property("float")
            plt.title("last move was captured")
            vprop[game.board.board_index_to_vertex[response]] = 1
        return vprop

    return evaluater

def maker_breaker_evaluater(maker,breaker):
    ev_maker = model_to_evaluater(maker)
    ev_breaker = model_to_evaluater(breaker)
    def maker_breaker_ev(game,respond_to=None):
        if game.view.gp["m"]:
            return ev_maker(game)
        else:
            return ev_breaker(game)
    return maker_breaker_ev


def interactive_hex_window(size, model_player=None, model_evaluater=None):
    global manual_mode,game,show_dead_and_captured,action_history,show_graph,is_over
    print(
"""
Instructions:
r: restart              left:  backwards
m: toggle ai moves      space: play ai move
s: switch colors        e:     show evaluation
a: toggle graph         g:     show graph
n: show graph assoc     q:     quit
k: hide graph
c: toggle show dead and captured""", end="")
    
    os.system("pkill mupdf")
    plt.rcParams['keymap.save'].remove('s')
    game_history = []
    manual_mode = True
    show_dead_and_captured = True
    show_graph = False
    is_over = False

    action_history = [None]

    xstart = -(size//2)*1.5
    ystart = -(size/2*np.sqrt(3/4))+0.5
    coords = []
    for i in range(size):
        for j in range(size):
            coords.append([xstart+0.5*j+i,ystart+np.sqrt(3/4)*j])
    coords = np.array(coords)

    def place_stone(action):
        global is_over
        game_history.append(game.copy())
        result = game.board.make_move(action,remove_dead_and_captured=True)
        fig.axes[0].cla()
        game.board.matplotlib_me(fig=fig)
        winner = game.who_won()
        is_over = winner is not None
        if winner is not None:
            if winner == "m":
                plt.title("Maker(Red) won the game")
            else:
                plt.title("Breaker(Blue) won the game")
        plt.pause(0.001)
        if result == "only board":
            action_history.append(game.board.board_index_to_vertex[action])
        elif result == "legal":
            action_history.append(None)
            if show_graph:
                do_graph_show()
        return result

    def show_assoc():
        fig.axes[0].cla()
        game.board.matplotlib_me(fig=fig,vprop=game.view.vertex_index,color_based_on_vprop=False)
        plt.pause(0.001)


    def show_eval():
        if model_evaluater is not None:
            fig.axes[0].cla()
            vprop = model_evaluater(game,respond_to=action_history[-1])
            game.board.matplotlib_me(fig=fig,vprop=vprop)
            plt.pause(0.001)

    def do_graph_show():
        game.draw_me(fname="hex_gui_graph.pdf",vprop1=game.view.vertex_index)
        os.system("pkill mupdf")
        os.system("nohup mupdf hex_gui_graph.pdf > /dev/null 2>&1 &")
        time.sleep(0.1)
        os.system("bspc node -f west")

    def on_press(event):
        global manual_mode, game,show_dead_and_captured,action_history,show_graph,is_over
        if event.key == "m":
            manual_mode = not manual_mode
        
        elif event.key == "n":
            show_assoc()

        elif event.key == "a":
            show_graph = not show_graph
            if show_graph:
                do_graph_show()
            else:
                os.system("pkill mupdf")

        elif event.key == "q":
            os.system("pkill mupdf")
            exit()

        elif event.key == "g":
            do_graph_show()

        elif event.key == "k":
            os.system("pkill mupdf")

        elif event.key == "c":
            show_dead_and_captured = not show_dead_and_captured
            game.callback_everything = show_dead_and_captured
            if show_dead_and_captured:
                game.board.fill_dead_and_captured()
                fig.axes[0].cla()
                game.board.matplotlib_me(fig=fig)
                plt.pause(0.001)
        elif event.key == " " and model_player is not None and not is_over:
            action = model_player(game,respond_to=action_history[-1])
            place_stone(action)

        elif event.key == "left":
            is_over = False
            if len(game_history)>0:
                game = game_history.pop()
                action_history.pop()
                plt.title("")
                fig.axes[0].cla()
                game.board.matplotlib_me(fig=fig)
                game.callback_everything = show_dead_and_captured
                plt.pause(0.001)
            if show_graph:
                do_graph_show()

        elif event.key == "r":
            is_over=False
            action_history = [None]
            game = Hex_game(size)
            game.board_callback = game.board.graph_callback
            game.callback_everything = show_dead_and_captured
            plt.title("")
            fig.axes[0].cla()
            game.board.matplotlib_me(fig=fig)
            plt.pause(0.001)
            if show_graph:
                do_graph_show()
        elif event.key == "e":
            show_eval()
        elif event.key == "s":
            game.view.gp["m"] = not game.view.gp["m"]
            game.board.onturn = "r" if game.board.onturn=="b" else "b"

            

    def onclick(event):
        # if is_over:
            # return
        plt.title("")
        if not hasattr(event,"xdata") or event.xdata is None:
            return
        click_coord = np.array([event.xdata, event.ydata])
        distances = np.sum((coords-click_coord)**2,axis=1)
        to_place = np.argmin(distances)
        result = place_stone(to_place)
        if model_player is not None and not manual_mode and result!="illegal":
            action = model_player(game,respond_to=action_history[-1])
            place_stone(action)

    game = Hex_game(size)
    game.board_callback = game.board.graph_callback
    fig = game.board.matplotlib_me()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid = fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show()

if __name__ == "__main__":
    player_model = random_mover_model
    evaluater = random_evaluater
    interactive_hex_window(8,player_model,evaluater)


