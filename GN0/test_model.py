from GN0.model_frontend import evaluate_graph,evaluate_node_switching_game_state, evaluate_winpattern_game_state
from GN0.convert_graph import convert_node_switching_game_back
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.models import GCN, GraphSAGE
import os
import time
from GN0.generate_training_data import generate_winpattern_game_graphs, generate_hex_graphs
from GN0.models import GCN_with_glob, CachedGraphNorm, cachify_gnn
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
from torch.utils.data import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_random_graphs(model):
    dataset = SupervisedDataset(root='./data/olddata', device=device, pre_transform=hex_pre_transform,num_graphs=100,game_type="hex",drop=True,game_size=11)
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for i,batch in enumerate(loader):
            data_list = [hex_pre_transform(batch.get_example(i)).to(device) for i in range(batch.num_graphs)]
            # small_data = generate_hex_graphs(games_to_play=1,drop=True,game_size=11)[0]
            # data_list[0] = hex_pre_transform(small_data).to(device)
            batch = Batch.from_data_list(data_list)

            tt_mask = batch.train_mask | batch.test_mask
            # tt_mask = torch.ones(batch.y.size(0),dtype=bool)
            out = model(batch.x,batch.edge_index,set_cache=True)
            loss = F.mse_loss(out[tt_mask],batch.y[tt_mask])
            # return model.export_norm_cache()
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
                return



def eval(model,categorical=False):
    dataset = SupervisedDataset(root='./data/smalldata', device=device, pre_transform=hex_pre_transform,num_graphs=100,game_type="hex",drop=True,game_size=6)
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

def test_node_switching_model(model,drop=False):
    game = Hex_game(8)
    game.board_callback = game.board.graph_callback
    win = False
    letters = "abcdefghijklmnopqrstuvwxyz"
    while not win:
        voltprop,value = game.compute_node_voltages_exact()
        if drop:  
            dropprop = game.compute_node_currents(voltprop)
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
        move_str = input()
        os.system("pkill mupdf")
        if move_str=="":
            move = random.choice(actions)
        else:
            move = game.board.board_index_to_vertex[letters.index(move_str[0])+(int(move_str[1:])-1)*game.board.size]
        game.make_move(move,remove_dead_and_captured=True)
        game.prune_irrelevant_subgraphs()
