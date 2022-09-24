import matplotlib.pyplot as plt
from graph_game.hex_board_game import build_hex_grid
import numpy as np
from graph_game.graph_tools_games import Hex_game
from graph_game.shannon_node_switching_game import Node_switching_game
from GN0.convert_graph import convert_node_switching_game
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        print(respond_to)
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
    global manual_mode,game,show_dead_and_captured,action_history
    plt.rcParams['keymap.save'].remove('s')
    game_history = []
    manual_mode = True
    show_dead_and_captured = True
    action_history = [None]

    xstart = -(size//2)*1.5
    ystart = -(size/2*np.sqrt(3/4))+0.5
    coords = []
    for i in range(size):
        for j in range(size):
            coords.append([xstart+0.5*j+i,ystart+np.sqrt(3/4)*j])
    coords = np.array(coords)

    def place_stone(action):
        game_history.append(game.copy())
        result = game.board.make_move(action,remove_dead_and_captured=True)
        fig.axes[0].cla()
        game.board.matplotlib_me(fig=fig)
        winner = game.who_won()
        if winner is not None:
            if winner == "m":
                plt.title("Maker(Red) won the game")
            else:
                plt.title("Breaker(Blue) won the game")
        plt.pause(0.001)
        print("setting last action to ",game.board.board_index_to_vertex[action])
        if result == "only board":
            action_history.append(game.board.board_index_to_vertex[action])
        elif result == "legal":
            action_history.append(None)

    def show_eval():
        if model_evaluater is not None:
            fig.axes[0].cla()
            vprop = model_evaluater(game,respond_to=action_history[-1])
            game.board.matplotlib_me(fig=fig,vprop=vprop)
            plt.pause(0.001)

    def on_press(event):
        global manual_mode, game,show_dead_and_captured,action_history
        if event.key == "m":
            manual_mode = not manual_mode
        elif event.key == "c":
            show_dead_and_captured = not show_dead_and_captured
            game.callback_everything = show_dead_and_captured
        elif event.key == " " and model_player is not None:
            action = model_player(game,respond_to=action_history[-1])
            place_stone(action)
        elif event.key == "left":
            if len(game_history)>0:
                game = game_history.pop()
                action_history.pop()
                plt.title("")
                fig.axes[0].cla()
                game.board.matplotlib_me(fig=fig)
                game.callback_everything = show_dead_and_captured
                plt.pause(0.001)

        elif event.key == "r":
            action_history = [None]
            game = Hex_game(size)
            game.board_callback = game.board.graph_callback
            game.callback_everything = show_dead_and_captured
            plt.title("")
            fig.axes[0].cla()
            game.board.matplotlib_me(fig=fig)
            plt.pause(0.001)
        elif event.key == "e":
            show_eval()
        elif event.key == "s":
            game.view.gp["m"] = not game.view.gp["m"]
            game.board.onturn = "r" if game.board.onturn=="b" else "b"

            

    def onclick(event):
        plt.title("")
        if not hasattr(event,"xdata") or event.xdata is None:
            return
        click_coord = np.array([event.xdata, event.ydata])
        distances = np.sum((coords-click_coord)**2,axis=1)
        to_place = np.argmin(distances)
        place_stone(to_place)
        if model_player is not None and not manual_mode:
            action = model_player(game,respond_to=action_history[-1])
            place_stone(action)

    game = Hex_game(size)
    game.board_callback = game.board.graph_callback
    fig = game.board.matplotlib_me()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid = fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show()
