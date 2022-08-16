from GN0.model_frontend import evaluate_graph,evaluate_node_switching_game_state, evaluate_winpattern_game_state
from GN0.convert_graph import convert_node_switching_game_back
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.models import GCN, GraphSAGE
import os
import time
from GN0.generate_training_data import generate_winpattern_game_graphs, generate_hex_graphs
from GCN import GCN_with_glob
import torch
from graph_game.winpattern_game import Winpattern_game,Graph_Store
from graph_game.graph_tools_games import Qango6x6,Hex_game
from graph_game.shannon_node_switching_game import Node_switching_game
from GN0.graph_dataset import SupervisedDataset, hex_pre_transform
from tqdm import tqdm, trange
import random
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
from torch_geometric.data import Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_random_graphs(model):
    dataset = SupervisedDataset(root='./data/testdata', device=device, pre_transform=hex_pre_transform,num_graphs=100,game_type="hex",drop=True,game_size=11)
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for batch in loader:
            data_list = [hex_pre_transform(batch.get_example(i)).to(device) for i in range(batch.num_graphs)]
            small_data = generate_hex_graphs(games_to_play=1,drop=True,game_size=11)[0]
            data_list[0] = hex_pre_transform(small_data).to(device)
            batch = Batch.from_data_list(data_list)

            tt_mask = batch.train_mask | batch.test_mask
            # tt_mask = torch.ones(batch.y.size(0),dtype=bool)
            out = model(batch.x,batch.edge_index)
            loss = F.mse_loss(out[tt_mask],batch.y[tt_mask])
            print(loss)
            for i in range(batch_size):
                data = batch.get_example(i)
                print(len(data.x))
                print(data)
                graph,tprop = convert_node_switching_game_back(data)
                predprop = graph.new_vertex_property("double")
                predprop.a = out[batch.batch==i].cpu().numpy()[:,0]
                game = Node_switching_game.from_graph(graph)
                game.draw_me("pred_compare.pdf",tprop,predprop)
                os.system("nohup mupdf pred_compare.pdf > /dev/null 2>&1 &")
                time.sleep(0.1)
                os.system("bspc node -f west")
                try:
                    input()
                except KeyboardInterrupt:
                    os.system("pkill mupdf")
                    exit()

                os.system("pkill mupdf")



def eval(model,categorical=False):
    dataset = SupervisedDataset(root='./data/testdata', device=device, pre_transform=hex_pre_transform,num_graphs=100,game_type="hex",drop=True,game_size=11)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    acc_accumulate = Accuracy().to(device)
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch in tqdm(loader):
            tt_mask = batch.train_mask | batch.test_mask
            out = model(batch.x,batch.edge_index)
            if categorical:
                loss = F.binary_cross_entropy(out[tt_mask], batch.y[tt_mask])
                accuracy = acc_accumulate(out[tt_mask].flatten(), batch.y[tt_mask].flatten().long())
                accuracies.append(accuracy)
            else:
                loss = F.mse_loss(out[tt_mask],batch.y[tt_mask])
            losses.append(loss)
    if categorical:
        print("Testing accuracy:", sum(accuracies) / len(accuracies))
    print("Testing loss:", sum(losses) / len(losses))
    return sum(losses) / len(losses)

def test_node_switching_model(drop=False):
    model_hyperparams = dict(in_channels=3,out_channels=1,num_layers=13,hidden_channels=32,norm=GraphNorm(32),act="relu")
    
    model = GraphSAGE(**model_hyperparams).to(device)
    model.load_state_dict(torch.load("model/GraphSAGE_hex_best.pt",map_location=device)["model_state_dict"])
    model.eval()

    game = Hex_game(6)
    game.board_callback = game.board.graph_callback
    win = False
    while not win:
        voltprop = game.compute_node_voltages_exact()
        if drop:
            dropprop = game.compute_voltage_drops(voltprop)
            prop = dropprop
        else:
            prop = voltprop
        evaluate_node_switching_game_state(model,game,prop,device=device)
        print(game.board.draw_me())
        game.draw_me("pred_compare.pdf",prop,game.view.vp.p)
        os.system("nohup mupdf pred_compare.pdf > /dev/null 2>&1 &")
        time.sleep(0.1)
        os.system("bspc node -f west")
        actions = game.get_actions()
        choice = input()
        os.system("pkill mupdf")
        if choice=="":
            move = random.choice(actions)
        else:
            move = int(choice)
        game.make_move(move,remove_dead_and_captured=True)
        game.prune_irrelevant_subgraphs()



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
    model_hyperparams = dict(in_channels=3,out_channels=1,num_layers=13,hidden_channels=32,norm=GraphNorm(32),act="relu")
    model = GraphSAGE(**model_hyperparams).to(device)
    model.load_state_dict(torch.load("model/final_GraphSAGE_hex.pt",map_location=device)["model_state_dict"])
    model.eval()
    test_random_graphs(model)
    # eval(model,categorical=False)
    # test_node_switching_model(drop=True)
    # test_winpattern_model(1)
