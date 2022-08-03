from GN0.generate_training_data import generate_graphs
from GN0.convert_graph import convert_graph,convert_graph_back
from graph_game.winpattern_game import Winpattern_game
from utils.unite_pdfs import unite_pdfs
import os

basepath = os.path.abspath(os.path.dirname(__file__))

def delstuff(all=False):
    for f in os.listdir(basepath):
        if f.endswith(".pdf") and (f != "united.pdf" or all):
            os.remove(os.path.join(basepath,f))

def test_generate_training_data():
    delstuff(all=True)
    graphs = generate_graphs(1)
    for i,graph in enumerate(graphs):
        new_graph = convert_graph_back(graph)
        new_game:Winpattern_game = Winpattern_game.from_graph(new_graph)
        new_game.draw_me(i)
    unite_pdfs(basepath,os.path.join(basepath,"united.pdf"))
    delstuff()


if __name__ == '__main__':
    test_generate_training_data()