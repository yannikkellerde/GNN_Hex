from GN0.model_frontend import evaluate_graph,evaluate_game_state
from GN0.generate_training_data import generate_graphs
from GCN import GCN
import torch
from graph_game.graph_tools_game import Graph_game,Graph_Store
from graph_game.graph_tools_games import Qango6x6
from GN0.graph_dataset import SupervisedDataset,pre_transform
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(3,2,conv_layers=8,conv_dim=16,global_dim=16).to(device)

model.load_state_dict(torch.load("model/GCN_model.pt"))
model.eval()

def eval(model):
    dataset = SupervisedDataset(root='./data/test_dataset', device=device, pre_transform=pre_transform,num_graphs=100)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    acc_accumulate = Accuracy().to(device)
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch in tqdm(loader):
            tt_mask = batch.train_mask | batch.test_mask
            out = model(batch)
            loss = F.binary_cross_entropy(out[tt_mask], batch.y[tt_mask])
            accuracy = acc_accumulate(out[tt_mask].flatten(), batch.y[tt_mask].flatten().long())
            accuracies.append(accuracy)
            losses.append(loss)
    print("Testing accuracy:", sum(accuracies) / len(accuracies))
    print("Testing loss:", sum(losses) / len(losses))
    return sum(accuracies) / len(accuracies)


def test_model(games_to_play):
    """ 
    Args:
        games_to_play: Number of games to play.
    """
    def reload(game:Graph_game,storage:Graph_Store):
        game.load_storage(storage)
        iswin = game.graph.new_vertex_property("vector<bool>")
        game.graph.vp.w = iswin
        for v in game.graph.vertices():
            game.graph.vp.w[v] = [False] * 2
    game = Qango6x6()
    start_pos = list("ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff"
                     "ffffff")
    game.board.position = start_pos
    game.graph_from_board()
    iswin = game.graph.new_vertex_property("vector<bool>")
    # 1: Is win for the player to move by forced moves
    # 2: Is win for the player not to move by forced moves

    game.graph.vp.w = iswin
    for v in game.graph.vertices():
        game.graph.vp.w[v] = [False] * 2
    start_storage = game.extract_storage()
    graphs = []
    known_hashes = set()
    for _ in range(games_to_play):
        win = False
        while 1:
            actions = game.get_actions(filter_superseeded=False,none_for_win=False)
            if len(actions) == 0:
                break
            move = random.choice(actions)
            win = game.make_move(move)
            game.board.position = game.board.pos_from_graph()
            game.hashme()
            if win:
                break
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
            evaluate_game_state(model,game,device=device)
            pred_model = game.board.draw_me_with_prediction(game.view.vp.p)
            pred_gt = game.board.draw_me_with_prediction(game.view.vp.w)
            print(pred_model)
            print(pred_gt)
            input()

        reload(game,start_storage)

if __name__ == "__main__":
    #eval(model)
    test_model(10)