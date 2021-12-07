from GN0.convert_graph import convert_graph
from graph_game.graph_tools_games import Qango6x6

def test_convert_graph():
    game = Qango6x6()
    start_pos = list("ffffff"
                     "ffwfff"
                     "fffbff"
                     "ffffff"
                     "fbffwf"
                     "ffffff")
    game.board.position = start_pos
    game.graph_from_board()
    game.make_move(7)
    win = game.graph.new_vertex_property("vector<bool>")
    game.view.vp.w = win
    for v in game.graph.vertices():
        win[v] = [False] * 2
    graph,vertexmap = convert_graph(game.view)
    print(vertexmap)
    print([(x.item(),y.item()) for x,y in zip(*graph.edge_index)])
    game.draw_me()

if __name__ == "__main__":
    test_convert_graph()