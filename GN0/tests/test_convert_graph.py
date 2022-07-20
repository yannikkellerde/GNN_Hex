from GN0.convert_graph import convert_graph,convert_graph_back
from graph_game.graph_tools_games import Qango6x6
from graph_game.graph_tools_game import Graph_game
from torch_geometric.utils import to_networkx
from GN0.util import visualize_graph

def test_convert_graph():
    """Checks convert_graph, convert_graph_back cycle consistency."""
    game = Qango6x6()
    start_pos = list("ffffff"
                     "ffwfff"
                     "fffbff"
                     "ffffff"
                     "fbffwf"
                     "ffffff")
    game.board.position = start_pos
    game.board.graph_from_board()
    game.make_move(7)
    game.make_move(14)
    game.view.gp["b"] = True
    win = game.graph.new_vertex_property("vector<bool>")
    game.view.vp.w = win
    for v in game.graph.vertices():
        win[v] = [False] * 2
    geometric_graph,vertexmap = convert_graph(game.view)
    game.draw_me(0)
    new_graph = convert_graph_back(geometric_graph)
    new_game:Graph_game = Graph_game.from_graph(new_graph)
    new_game.draw_me(1)
    new_game.hashme()
    game.hashme()
    assert game.hash == new_game.hash

if __name__ == "__main__":
    test_convert_graph()
