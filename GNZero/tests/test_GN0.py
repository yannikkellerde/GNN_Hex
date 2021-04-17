import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))

from GNZero.GN0 import GN0
from game.graph_tools_games import Tic_tac_toe
import tensorflow as tf


def test_GN():
    game = Tic_tac_toe()
    gn0 = GN0(game)
    res = gn0.model.do_policy_and_value(game.view)
    print(res)

if __name__ == "__main__":
    test_GN()