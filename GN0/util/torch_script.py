from GN0.models import PV_torch_script
from graph_game.graph_tools_games import Hex_game
from GN0.util.convert_graph import convert_node_switching_game
import torch

def trace_pv_model():
    model = PV_torch_script(hidden_channels=25,hidden_layers=10,policy_layers=2,value_layers=2,in_channels=3)
    node_features = torch.ones((5,3))
    edge_index = torch.empty((2,2),dtype=torch.long);
    graph_indices = torch.zeros(5,dtype=torch.long);
    edge_index[0][0] = 0
    edge_index[0][1] = 3
    edge_index[1][0] = 3
    edge_index[1][1] = 4
    print(model(node_features,edge_index,graph_indices))

    
    traced = torch.jit.script(model)
    traced.save("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/alpha_zero/saved_models/traced.pt")

def test_traced_model(path):
    game = Hex_game(8)
    data = convert_node_switching_game(game.view,global_input_properties=[int(game.view.gp["m"])])
    loaded = torch.jit.load(path)
    res = loaded(data.x,data.edge_index,data.x.new_zeros(data.x.size(0),dtype=torch.long))
    print(res)

if __name__ == "__main__":
    trace_pv_model()
    # test_traced_model("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/alpha_zero/saved_models/traced.pt")
