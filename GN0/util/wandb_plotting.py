import wandb
import pandas as pd 
import matplotlib.pyplot as plt
import os
basepath = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(basepath,"../../csv")
img_path = os.path.join(basepath,"../../images/wandb_plots")

def get_run_df(name="project.csv",run_name="icv7ozhh"):
    api = wandb.Api()
    run = api.run(f"yannikkellerde/rainbow_hex/{run_name}")
    hist = run.scan_history()
    df = pd.DataFrame(hist)
    df.to_csv(os.path.join(csv_path,name))
    
def runtime_game_frame():
    cnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"fully_cnn_7x7.csv"))
    gnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"gnn_7x7.csv"))
    cnn_df = cnn_df[cnn_df["_runtime"]<=gnn_df["_runtime"].max()]
    plt.plot(cnn_df["_runtime"]/3600,cnn_df["game_frame"]/1000000,label="cnn",color="blue")
    plt.plot(gnn_df["_runtime"]/3600,gnn_df["game_frame"]/1000000,label="gnn",color="orange")
    plt.xlabel("runtime (hours)")
    plt.ylabel("moves made (millions)")
    plt.savefig(os.path.join(img_path,"runtime_game_frame.svg"))
    plt.show()

def vs_random_winrate():
    cnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"fully_cnn_7x7.csv"))
    gnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"gnn_7x7.csv"))
    cnn_df = cnn_df[cnn_df["_runtime"]<=gnn_df["_runtime"].max()]
    cnn_df.dropna(subset=["ev/maker_random_winrate","ev/breaker_random_winrate"],inplace=True)
    gnn_df.dropna(subset=["ev/maker_random_winrate","ev/breaker_random_winrate"],inplace=True)
    cnn_df["mean_random_winrate"] = (cnn_df["ev/maker_random_winrate"]+cnn_df["ev/breaker_random_winrate"])/2
    gnn_df["mean_random_winrate"] = (gnn_df["ev/maker_random_winrate"]+gnn_df["ev/breaker_random_winrate"])/2
    cnn_df["rolling"] = cnn_df["mean_random_winrate"].rolling(10).mean()
    gnn_df["rolling"] = gnn_df["mean_random_winrate"].rolling(10).mean()
    plt.plot(cnn_df["_runtime"]/3600,cnn_df["rolling"],color="blue",label="cnn")
    plt.plot(gnn_df["_runtime"]/3600,gnn_df["rolling"],color="orange",label="gnn")
    plt.plot(cnn_df["_runtime"]/3600,cnn_df["mean_random_winrate"],color="blue",alpha=0.2)
    plt.plot(gnn_df["_runtime"]/3600,gnn_df["mean_random_winrate"],color="orange",alpha=0.2)
    plt.xlabel("runtime (hours)")
    plt.ylabel("winrate vs random")
    plt.legend()
    plt.savefig(os.path.join(img_path,"vs_random.svg"))
    plt.show()


def losses():
    cnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"fully_cnn_7x7.csv"))
    gnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"gnn_7x7.csv"))
    cnn_df = cnn_df[cnn_df["_runtime"]<=gnn_df["_runtime"].max()]


    cnn_df.dropna(subset=["maker/losses","breaker/losses"],inplace=True)
    gnn_df.dropna(subset=["maker/losses","breaker/losses"],inplace=True)
    cnn_df["mean_loss"] = (cnn_df["maker/losses"]+cnn_df["breaker/losses"])/2
    gnn_df["mean_loss"] = (gnn_df["maker/losses"]+gnn_df["breaker/losses"])/2

    cnn_df = cnn_df[cnn_df["mean_loss"]<=0.015]
    gnn_df = gnn_df[gnn_df["mean_loss"]<=0.015]

    cnn_df["rolling"] = cnn_df["mean_loss"].rolling(100).mean()
    gnn_df["rolling"] = gnn_df["mean_loss"].rolling(100).mean()
    plt.plot(cnn_df["_runtime"]/3600,cnn_df["rolling"],color="blue",label="cnn")
    plt.plot(gnn_df["_runtime"]/3600,gnn_df["rolling"],color="orange",label="gnn")
    plt.plot(cnn_df["_runtime"]/3600,cnn_df["mean_loss"],color="blue",alpha=0.2)
    plt.plot(gnn_df["_runtime"]/3600,gnn_df["mean_loss"],color="orange",alpha=0.2)
    plt.xlabel("runtime (hours)")
    plt.ylabel("loss (temporal difference MSE)")
    plt.legend()
    plt.savefig(os.path.join(img_path,"losses.svg"))
    plt.show()

if __name__ == "__main__":
    # get_run_df(name="gnn_7x7.csv",run_name="x3qias30")
    # get_run_df(name="fully_cnn_7x7.csv",run_name="icv7ozhh")
    # get_big_df()
    vs_random_winrate()
    losses()
    runtime_game_frame()
