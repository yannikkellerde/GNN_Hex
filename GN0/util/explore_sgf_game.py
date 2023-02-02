import matplotlib.pyplot as plt
from graph_game.graph_tools_games import Hex_game
import sys


def interactive_hex_window(sgf_file):
    global game, board_moves, game_history, cur_idx, fig
    plt.rcParams["keymap.yscale"].remove('l')
    plt.rcParams['keymap.save'].remove('s')
    with open(sgf_file,"r") as f:
        stuff = f.read().replace("\n","").replace("\r","").split(";")
    hex_size = int(stuff[1].split("SZ[")[1].split("]")[0])
    game = Hex_game(hex_size)
    game.board_callback = game.board.graph_callback
    board_moves = [game.board.notation_to_number(x.split("[")[1].split("]")[0]) for x in stuff[2:]]
    starting_player = stuff[2][1]
    if starting_player=="W":
        game.view.gp["m"] = True
    else:
        game.view.gp["m"] = False
    game_history = [game.copy()]
    cur_idx = 0

    fig = game.board.matplotlib_me()
    cid = fig.canvas.mpl_connect('key_press_event', onpress)
    plt.show()

def onpress(event):
    global cur_idx, game
    print(event.key)
    if event.key == "right":
        if cur_idx<len(board_moves):
            vertex = game.board.board_index_to_vertex_index[board_moves[cur_idx]]
            print(vertex)
            game.make_move(vertex, remove_dead_and_captured=True)
            game_history.append(game.copy())
            cur_idx+=1;
            game.board.matplotlib_me(fig=fig)
    elif event.key == "left":
        if cur_idx>0:
            cur_idx-=1
            game = game_history[cur_idx]
            game.board.matplotlib_me(fig=fig)

if __name__ == "__main__":
    interactive_hex_window(sys.argv[1])
