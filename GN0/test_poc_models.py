from GN0.models import GCN_with_glob
from torch_geometric.nn.models import GCN, GraphSAGE
from GN0.model_frontend import evaluate_winpattern_game_state
from graph_game.winpattern_game import Winpattern_game,Graph_Store
from graph_game.graph_tools_games import Qango6x6
from GN0.test_model import test_node_switching_model
import sys
from GN0.models import GCN_with_glob, CachedGraphNorm, cachify_gnn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_winpattern_model(games_to_play):
    """ 
    Args:
        games_to_play: Number of games to play.
    """
    def reload(game:Winpattern_game,storage:Graph_Store):
        game.load_storage(storage)
        iswin = game.graph.new_vertex_property("vector<bool>")
        game.graph.vp.w = iswin
        for v in game.graph.vertices():
            game.graph.vp.w[v] = [False] * 2

    model = GCN_with_glob(3,2,conv_layers=8,conv_dim=16,global_dim=16).to(device)

    model.load_state_dict(torch.load("model/GCN_model.pt",map_location=device))
    model.eval()
    game = Qango6x6()
    start_pos = list("ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff")
    game.board.position = start_pos
    game.board.graph_from_board()
    iswin = game.graph.new_vertex_property("vector<bool>")
    # 1: Is win for the player to move by forced moves
    # 2: Is win for the player not to move by forced moves

    game.graph.vp.w = iswin
    for v in game.graph.vertices():
        game.graph.vp.w[v] = [False] * 2
    start_storage = game.extract_storage()
    graphs = []
    known_hashes = set()
    letters = "abcdef"
    for _ in range(games_to_play):
        win = False
        while 1:
            moves = game.get_actions(filter_superseeded=False,none_for_win=False)
            if len(moves) == 0:
                break
            for move in moves:
                game.view.vertex(move)
            storage = game.extract_storage()
            evals = game.check_move_val(moves,priorize_sets=False)
            game.load_storage(storage)
            for move,ev in zip(moves,evals):
                if (ev in [-3,-4] and game.onturn=="w") or (ev in [3,4] and game.onturn=="b"):
                    game.graph.vp.w[game.view.vertex(move)] = [True,False]
                elif (ev in [-3,-4] and game.onturn=="b") or (ev in [3,4] and game.onturn=="w"):
                    game.graph.vp.w[game.view.vertex(move)] = [False,True]
                else:
                    game.graph.vp.w[game.view.vertex(move)] = [False,False]
            evaluate_winpattern_game_state(model,game,device=device)
            pred_model = game.board.draw_me_with_prediction(game.view.vp.p)
            pred_gt = game.board.draw_me_with_prediction(game.view.vp.w)
            model_rows = pred_model.split("\n")
            gt_rows = pred_gt.split("\n")
            new_rows = [m+"    "+g for m,g in zip(model_rows,gt_rows)]
            print("   pred          gt  ")
            print("\n".join(new_rows))
            move_choice = input()
            actions = game.get_actions(filter_superseeded=False,none_for_win=False)
            if len(actions) == 0:
                break
            if move_choice=="":
                move = random.choice(actions)
            else:
                board_move = (int(move_choice[1])-1) * len(letters) + letters.index(move_choice[0])
                move = game.board.node_map_rev[board_move]
                if move not in actions:
                    print(f"Invalid move {move}. Not in moves {actions}")
                    continue

            win = game.make_move(move)
            game.board.position = game.board.pos_from_graph()
            game.hashme()
            if win:
                break

        reload(game,start_storage)


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "qango":
        test_winpattern_model(1)
    else:
        model_hyperparams = dict(in_channels=3,out_channels=1,num_layers=13,hidden_channels=32,norm=CachedGraphNorm(32),act="relu")
        model = cachify_gnn(GraphSAGE)(**model_hyperparams).to(device)
        state_dict = torch.load("model/GraphSAGE_unnormalized.pt",map_location=device)
        model.load_state_dict(state_dict["model_state_dict"])
        model.import_norm_cache(*state_dict["cache"])
        test_node_switching_model(model,drop=True)
