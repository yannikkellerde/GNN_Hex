from GN0.convert_graph import convert_winpattern_game, convert_node_switching_game
from graph_game.winpattern_game import Winpattern_game
import torch
from graph_tool.all import Graph, VertexPropertyMap
from typing import Callable
from graph_game.shannon_node_switching_game import Node_switching_game
from graph_game.utils import get_view_index_map


def evaluate_graph(model:torch.nn.Module, graph:Graph, conversion_func:Callable, device="cpu", categorical=False, full_data=False):
    """Evaluate a graph using a trained model.
    
    Args:
        model: A trained model
        graph: A graph-tool graph

    """
    graph_data = conversion_func(graph)
    vertexmap = get_view_index_map(graph)
    graph_data.x = graph_data.x.float()
    graph_data.edge_index = graph_data.edge_index
    graph_data.to(device)
    model.eval()
    with torch.no_grad():
        if full_data:
            output = model(graph_data)
        else:
            output = model(graph_data.x,graph_data.edge_index)
    if categorical:
        pred_map = graph.new_vertex_property("vector<bool>")
    else:
        pred_map = graph.new_vertex_property("double")
    graph.vp.p = pred_map
    for ind,pred in enumerate(output):
        if categorical:
            pred_map[graph.vertex(vertexmap[ind])] = (pred>0.5).cpu().numpy()
        else:
            pred_map[graph.vertex(vertexmap[ind])] = pred.cpu().numpy()

def evaluate_winpattern_game_state(model:torch.nn.Module,game:Winpattern_game,device="cpu"):
    return evaluate_graph(model,game.view,convert_winpattern_game,device=device,full_data=True,categorical=True)

def evaluate_node_switching_game_state(model:torch.nn.Module,game:Node_switching_game,target_prop_map:VertexPropertyMap,device="cpu"):
    return evaluate_graph(model,game.view,lambda x:convert_node_switching_game(x,target_prop_map),device=device)
