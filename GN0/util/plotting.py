from GN0.RainbowDQN.evaluate_elo import Elo_handler,random_player
import numpy as np
from graph_game.graph_tools_games import Hex_game
from GN0.RainbowDQN.Rainbow.common.utils import get_highest_model_path
import matplotlib.pyplot as plt

device = "cpu"

def plot_transfer_elos():
    e = Elo_handler(7,k=2,device=device)
    e.load_a_model_player(get_highest_model_path("gnn_7x7/7"),"modern_two_headed","gnn")
    e.load_a_model_player(get_highest_model_path("cnn_7x7_fully_conv/7"),"fully_conv","cnn",cnn_mode=True,cnn_hex_size=None)
    e.add_player(name="random",model=random_player,set_rating=0,uses_empty_model=False,simple=True,rating_fixed=True)

    gnn_win_percents=[]
    start = 5
    end = 14

    for hex_size in range(start,end):
        e.size = hex_size
        e.players["gnn"]["rating"]=0
        e.players["cnn"]["rating"]=0
        # e.roundrobin(3,None,must_include_players=["random","gnn","cnn"],score_as_n_games=5000)
        result1 = e.play_some_games("cnn","gnn",None,0,progress=True)
        result2 = e.play_some_games("gnn","cnn",None,0,progress=True)
        cnn_wins = result1["cnn"]+result2["cnn"]
        gnn_wins = result1["gnn"]+result2["gnn"]
        gnn_win_percents.append(gnn_wins/(cnn_wins+gnn_wins))

    gnn_win_percents = np.array(gnn_win_percents)*100
    plt.yticks((0,50,100))
    plt.xticks(np.arange(start,end))
    plt.ylabel("Win percent")
    plt.xlabel("Hex size")

    plt.bar(np.arange(start,end),gnn_win_percents,width=0.8,color="darkorange",label="GNN")
    plt.bar(np.arange(start,end),100-gnn_win_percents,width=0.8,bottom=gnn_win_percents, color="b",label="CNN")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_transfer_elos()



