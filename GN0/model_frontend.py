from GN0.convert_graph import convert_graph
from graph_game.winpattern_game import Winpattern_game
import torch
from graph_tool.all import Graph


def evaluate_graph(model:torch.nn.Module, graph:Graph,device="cpu"):
    """Evaluate a graph using a trained model.
    
    Args:
        model: A trained model
        graph: A graph-tool graph

    """
    graph_data,vertexmap = convert_graph(graph)
    graph_data.x = graph_data.x.float()
    graph_data.edge_index = graph_data.edge_index
    graph_data.to(device)
    model.eval()
    with torch.no_grad():
        output = model(graph_data)
    pred_map = graph.new_vertex_property("vector<bool>")
    graph.vp.p = pred_map
    for ind,pred in enumerate(output):
        #print(ind,pred,vertexmap[ind],pred>0.5)
        pred_map[graph.vertex(vertexmap[ind])] = (pred>0.5).cpu().numpy()

def evaluate_game_state(model:torch.nn.Module,game:Winpattern_game,device="cpu"):
    return evaluate_graph(model,game.view,device=device)