import Qango6x6 from graph_tools_games
from convert_graph import convert
import random

def generate_graphs(number_to_generate,batch_size,none_for_win):
    game = Qango6x6()
    start_pos = list("ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff")
    game.board.position = start_pos
    game.graph_from_board()
    iswin = game.graph.new_vertex_property("bool")
    game.graph.vp.w = iswin
    graphs = []
    data_batches = []
    for i in range(number_to_generate):
        win = False
        while not win:
            actions = game.get_actions(filter_superseeded=False,none_for_win=False)
            win = game.make_move(random.choice(actions))
            if win:
                game.board.position = start_pos
                game.graph_from_board()
                iswin = game.graph.new_vertex_property("bool")
                game.graph.vp.w = iswin
        winmoves = self.game.win_threat_search(one_is_enough=False,until_time=time.time()+2)
        for move in winmoves:
            game.view.vp.w[game.view.vertex(move)] = True
        if len(graphs)>=batch_size:
            data_batches.append(convert(graphs))
        graphs.append(game.view)
    if len(graphs)>0:
        data_batches.append(convert(graphs))
    return data_batches