from GN0.convert_graph import convert_graph
from graph_game.graph_tools_game import Graph_game
import torch
from graph_tool.all import Graph


def evaluate_graph(model:torch.nn.Module, graph:Graph):
    """Evaluate a graph using a trained model.
    
    Args:
        model: A trained model
        graph: A graph-tool graph

    """
    graph_data,vertexmap = convert_graph(graph)
    graph_data.x = graph_data.x.to(model.device)
    graph_data.edge_index = graph_data.edge_index.to(model.device)
    graph_data.y = graph_data.y.to(model.device)
    model.eval()
    with torch.no_grad():
        output = model(graph_data)
    pred_map = graph.new_vertex_property("vector<bool>")
    graph.vp.p = pred_map
    for pred,vert_ind in zip(output,vertexmap):
        pred_map[graph.vertex(vert_ind)] = pred

def evaluate_game_state(model:torch.nn.Module,game:Graph_game):
    return evaluate_graph(model,game.view)