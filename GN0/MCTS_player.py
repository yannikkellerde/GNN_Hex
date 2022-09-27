from GN0.MCTS import MCTS
from GN0.convert_graph import convert_node_switching_game
from graph_game.shannon_node_switching_game import Node_switching_game
import torch.nn.functional as F 

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
        
