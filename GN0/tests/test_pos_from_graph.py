from graph_game.graph_tools_games import Hex_game
from graph_game.hex_board_game import Hex_board
import matplotlib.pyplot as plt

def test_pos_from_graph():
    game = Hex_game(7)
    game.board.position = (
    "bbbrbbb"
    "rrbbrrr"
    "ffffrrr"
    "ffffbrb"
    "bfffffb"
    "bbfrffb"
    "bbbbffr"
    )
    game.board.position = (
    "brrbbbr"
    "brrrfff"
    "brrbfff"
    "rbfffrb"
    "bbffffb"
    "brfffbb"
    "brffbbb"
    )
    game.board.graph_from_board(True)
    game.board_callback = game.board.graph_callback
    # game.board.matplotlib_me()
    # plt.show()
    # game.board.position=[]
    # game.board.pos_from_graph()
    # game.board.matplotlib_me()
    # plt.show()
    game.draw_me("7x7.pdf")
    game_big = Hex_game.from_graph(game.graph)
    game_big.board = Hex_board()
    game_big.board.game = game_big
    game_big.board.size = 11
    game_big.board.squares = 11*11
    game_big.board.pos_from_graph(initialize=False)
    game.board.matplotlib_me()
    plt.show()
    game.draw_me("11x11.pdf")

if __name__ == "__main__":
    test_pos_from_graph()
