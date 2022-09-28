from GN0.MCTS import MCTS
from GN0.convert_graph import convert_node_switching_game
from graph_game.shannon_node_switching_game import Node_switching_game
import torch.nn.functional as F 
import numpy as np
from graph_game.hex_gui import interactive_hex_window
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_model_for_mcts(model,temperature):
    def eval_pos(game:Node_switching_game):
        data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])],need_backmap=True)
        value,advantage = model(data.x,data.edge_index,seperate=True)
        policy = F.softmax(advantage[2:]/temperature,dim=0)
        moves = [data.backmap[x] for x in range(2,len(advantage))]
        return moves,policy,(value+1)/2
    return eval_pos

def get_mcts_player(model,policy_temperature,final_temperature,runtime):
    mcts_nn = prepare_model_for_mcts(model,policy_temperature)
    def choose_move(game):
        mcts = MCTS(game,mcts_nn)
        mcts.run(max_time=runtime)
        moves,probs = mcts.extract_result(final_temperature)
        move = np.random.choice(moves,p=probs)
        return move
    return choose_move

def get_mcts_evaluater(model,policy_temperature,final_temperature,runtime):
    mcts_nn = prepare_model_for_mcts(model,policy_temperature)
    def evaluate(game):
        mcts = MCTS(game,mcts_nn)
        mcts.run(max_time=runtime)
        moves,probs = mcts.extract_result(final_temperature)
        return probs
    return evaluate
    

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

    player = get_mcts_player(model,1,0,1)
    evaluater = get_mcts_evaluater(model,1,0,1)
    interactive_hex_window(11,model_player=player,model_evaluater=evaluater)



if __name__=="__main__":

