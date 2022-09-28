from GN0.MCTS import MCTS,Node
from GN0.convert_graph import convert_node_switching_game
from graph_game.shannon_node_switching_game import Node_switching_game
import torch.nn.functional as F 
import numpy as np
from graph_game.hex_gui import interactive_hex_window,playerify_model,mixed_player
import torch
from Rainbow.common.utils import get_highest_model_path
from GN0.models import get_pre_defined

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_model_for_mcts(model,temperature):
    def eval_pos(game:Node_switching_game):
        data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True)
        value,advantage = model(data.x,data.edge_index,seperate=True)
        policy = F.softmax(advantage[2:]/temperature,dim=0).detach().numpy()
        moves = [int(data.backmap[x]) for x in range(2,len(advantage))]
        return moves,policy,(float(value)+1)/2
    return eval_pos

def get_mcts_player(model,policy_temperature,final_temperature,runtime):
    mcts_nn = prepare_model_for_mcts(model,policy_temperature)
    def choose_move(game:Node_switching_game,**_kwargs):
        mcts = MCTS(game.copy(withboard=False),mcts_nn)
        mcts.run(max_time=runtime)
        moves,probs = mcts.extract_result(final_temperature)
        move = np.random.choice(moves,p=probs)
        move_num = list(moves).index(move)
        print(move_num,mcts.root.priors[move_num],np.max(mcts.root.priors))
        print(list(zip(mcts.root.visits,mcts.root.priors)))
        move = game.board.vertex_index_to_board_index[move]
        return move
    return choose_move

def get_pre_defined_mcts_model(model_name="azure-snowball-157"):
    version = None
    path = get_highest_model_path("azure-snowball-157")
    if version is not None:
        path = os.path.join(os.path.dirname(path),f"checkpoint_{version}.pt")
    stuff = torch.load(path,map_location=device)
    args = stuff["args"]
    model = get_pre_defined("two_headed",args).to(device)

    model.load_state_dict(stuff["state_dict"])
    if "cache" in stuff and stuff["cache"] is not None:
        model.import_norm_cache(*stuff["cache"])
    model.eval()

    player = prepare_model_for_mcts(model,1)
    return player



def get_mcts_evaluater(model,policy_temperature,final_temperature,runtime):
    mcts_nn = prepare_model_for_mcts(model,policy_temperature)
    def evaluate(game:Node_switching_game,**_kwargs):
        mcts = MCTS(game.copy(withboard=False),mcts_nn)
        mcts.run(max_time=runtime)
        moves,probs = mcts.extract_result(final_temperature)
        return probs
    return evaluate
    
def mcts_vs_model():
    version = None
    path = get_highest_model_path("azure-snowball-157")
    if version is not None:
        path = os.path.join(os.path.dirname(path),f"checkpoint_{version}.pt")
    stuff = torch.load(path,map_location=device)
    args = stuff["args"]
    model = get_pre_defined("two_headed",args).to(device)

    model.load_state_dict(stuff["state_dict"])
    if "cache" in stuff and stuff["cache"] is not None:
        model.import_norm_cache(*stuff["cache"])
    model.eval()

    mcts_player = get_mcts_player(model,1,0,2)
    model_player = playerify_model(model)
    # player = mixed_player(mcts_player,model_player)
    player = mixed_player(mcts_player,model_player)

    evaluater = get_mcts_evaluater(model,1,0,2)
    interactive_hex_window(11,model_player=player,model_evaluater=evaluater)


def play_vs_mcts():
    version = None
    path = get_highest_model_path("azure-snowball-157")
    if version is not None:
        path = os.path.join(os.path.dirname(path),f"checkpoint_{version}.pt")
    stuff = torch.load(path,map_location=device)
    args = stuff["args"]
    model = get_pre_defined("two_headed",args).to(device)

    model.load_state_dict(stuff["state_dict"])
    if "cache" in stuff and stuff["cache"] is not None:
        model.import_norm_cache(*stuff["cache"])
    model.eval()

    player = get_mcts_player(model,0.2,0,10)
    evaluater = get_mcts_evaluater(model,0.2,0,10)
    interactive_hex_window(11,model_player=player,model_evaluater=evaluater)


if __name__=="__main__":
    play_vs_mcts()
    # mcts_vs_model()

