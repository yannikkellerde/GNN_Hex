from GN0.model_frontend import evaluate_graph,evaluate_node_switching_game_state, evaluate_winpattern_game_state
import scipy.special
import torch_geometric.utils
from GN0.convert_graph import convert_node_switching_game_back
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.models import GCN, GraphSAGE
from torch_geometric.nn.conv import SAGEConv
import os
import time
from GN0.generate_training_data import generate_winpattern_game_graphs, generate_hex_graphs
from GN0.models import GCN_with_glob, CachedGraphNorm, cachify_gnn, PolicyValueGNN
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
import torch_geometric.utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_random_graphs(model):
    dataset = SupervisedDataset(root='./data/policy_value', device=device, pre_transform=hex_pre_transform,num_graphs=100,game_type="hex",drop=True,game_size=11)
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for i,batch in enumerate(loader):
            # data_list = [hex_pre_transform(batch.get_example(i)).to(device) for i in range(batch.num_graphs)]
            # small_data = generate_hex_graphs(games_to_play=1,drop=True,game_size=11)[0]
            # data_list[0] = hex_pre_transform(small_data).to(device)
            # batch = Batch.from_data_list(data_list)

            tt_mask = batch.train_mask | batch.test_mask
            # tt_mask = torch.ones(batch.y.size(0),dtype=bool)
            policy,value = model(batch.x,batch.edge_index,graph_indices=batch.batch)
            policy = policy[tt_mask]
            y = batch.y[tt_mask]
            policy_mse = F.mse_loss(policy,y)
            value_mse = F.mse_loss(value.squeeze(),batch.global_y.squeeze())
            # return model.export_norm_cache()
            print(policy_mse,value_mse)
            for i in range(batch_size):
                data = batch.get_example(i)
                print(data.global_y,value[i])
                graph,tprop = convert_node_switching_game_back(data)
                tprop.a[2:] = tprop.a[2:]
                predprop = graph.new_vertex_property("double")
                predprop.a[2:] = policy[batch.batch[tt_mask]==i].cpu().numpy()[:,0]
                game = Node_switching_game.from_graph(graph)
                game.draw_me("pred_compare.pdf",tprop,predprop,decimal_places=0)
                os.system("nohup mupdf pred_compare.pdf > /dev/null 2>&1 &")
                time.sleep(0.1)
                os.system("bspc node -f west")
                try:
                    input()
                except KeyboardInterrupt:
                    os.system("pkill mupdf")
                    exit()

                os.system("pkill mupdf")



def evaluate(model,categorical=False,old=False):
    dataset = SupervisedDataset(root='./data/policy_value', device=device, pre_transform=hex_pre_transform,num_graphs=100,game_type="hex",drop=True,game_size=6)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    acc_accumulate = Accuracy().to(device)
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch in tqdm(loader):
            tt_mask = batch.train_mask | batch.test_mask
            if old:
                out = torch_geometric.utils.softmax(model(batch.x[:,:3],batch.edge_index),batch.batch)
            else:
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

if __name__ == "__main__":
    default_model_hyperparams = dict(in_channels=4,num_layers=13,hidden_channels=32,norm=CachedGraphNorm(32),act="relu",policy_head=SAGEConv)
    cached_model = cachify_gnn(GraphSAGE) 
    model = PolicyValueGNN(GNN=cached_model,**default_model_hyperparams).to(device)
    state_dict = torch.load("model/GraphSAGE_best.pt",map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.import_norm_cache(*state_dict["cache"])
    model.eval()
    test_random_graphs(model)
