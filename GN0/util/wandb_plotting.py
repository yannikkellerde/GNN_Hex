"""Uses the wandb api to create plots for the thesis

Key functions include:
    supervised_stuff: Create Figure 5.7
    curriculum_stuff: Create Figure 5.6 b
    losses: Create Figure 5.1 b
    vs_random_winrate: Create Figure 5.1 c
"""

import wandb
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import patches
import os
basepath = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(basepath,"../../csv")
img_path = os.path.join(basepath,"../../images/wandb_plots")

def supervised_stuff():
    os.makedirs(os.path.join(img_path,"supervised_compare"),exist_ok=True)

    cnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"supervised_cnn.csv"))
    gnn_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"supervised_gnn.csv"))
    gao_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"supervised_gao.csv"))
    cnn_df = cnn_df[:3000].dropna(subset=["policy_acc/train","policy_acc/val","loss/val"])
    gnn_df = gnn_df[:3000].dropna(subset=["policy_acc/train","policy_acc/val","loss/val"])
    gao_df = gao_df[:3000].dropna(subset=["policy_acc/train","policy_acc/val","loss/val"])

    cnn_df["policy_rolling_val"] = cnn_df["policy_acc/val"].rolling(20).mean()
    gnn_df["policy_rolling_val"] = gnn_df["policy_acc/val"].rolling(20).mean()
    cnn_df["policy_rolling_train"] = cnn_df["policy_acc/train"].rolling(20).mean()
    gnn_df["policy_rolling_train"] = gnn_df["policy_acc/train"].rolling(20).mean()
    cnn_df["value_rolling_val"] = cnn_df["value_acc_sign/val"].rolling(20).mean()
    gnn_df["value_rolling_val"] = gnn_df["value_acc_sign/val"].rolling(20).mean()
    cnn_df["value_rolling_train"] = cnn_df["value_acc_sign/train"].rolling(20).mean()
    gnn_df["value_rolling_train"] = gnn_df["value_acc_sign/train"].rolling(20).mean()
    cnn_df["loss_rolling_train"] = cnn_df["loss/train"].rolling(20).mean()
    gnn_df["loss_rolling_train"] = gnn_df["loss/train"].rolling(20).mean()
    cnn_df["loss_rolling_val"] = cnn_df["loss/val"].rolling(20).mean()
    gnn_df["loss_rolling_val"] = gnn_df["loss/val"].rolling(20).mean()

    gao_df["policy_rolling_val"] = gao_df["policy_acc/val"].rolling(20).mean()
    gao_df["policy_rolling_train"] = gao_df["policy_acc/train"].rolling(20).mean()
    gao_df["value_rolling_val"] = gao_df["value_acc_sign/val"].rolling(20).mean()
    gao_df["value_rolling_train"] = gao_df["value_acc_sign/train"].rolling(20).mean()
    gao_df["loss_rolling_train"] = gao_df["loss/train"].rolling(20).mean()
    gao_df["loss_rolling_val"] = gao_df["loss/val"].rolling(20).mean()

    plt.plot(gao_df.index,gao_df["policy_acc/train"],color="C2",alpha=0.2)
    plt.plot(gao_df.index,gao_df["policy_rolling_train"],color="C2",alpha=1,label="Gao")
    plt.plot(cnn_df.index,cnn_df["policy_acc/train"],color="C0",alpha=0.2)
    plt.plot(gnn_df.index,gnn_df["policy_acc/train"],color="C1",alpha=0.2)
    plt.plot(cnn_df.index,cnn_df["policy_rolling_train"],color="C0",alpha=1,label="U-Net")
    plt.plot(gnn_df.index,gnn_df["policy_rolling_train"],color="C1",alpha=1,label="GNN")
    plt.ylim(min(cnn_df["policy_acc/train"].min(),gnn_df["policy_acc/train"].min(),cnn_df["policy_acc/val"].min(),gnn_df["policy_acc/val"].min()))
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("training policy accuracy")
    # plt.savefig(os.path.join(img_path,"supervised_compare","training_policy_acc.svg"))
    # plt.clf()

    plt.plot(gao_df.index,gao_df["policy_acc/val"],color="C2",alpha=0.2)
    plt.plot(gao_df.index,gao_df["policy_rolling_val"],color="C2",alpha=1,linestyle="--",dashes=(5, 3))
    plt.plot(cnn_df.index,cnn_df["policy_acc/val"],color="C0",alpha=0.2)
    plt.plot(gnn_df.index,gnn_df["policy_acc/val"],color="C1",alpha=0.2)
    plt.plot(cnn_df.index,cnn_df["policy_rolling_val"],color="C0",alpha=1,linestyle="--",dashes=(5, 3))
    plt.plot(gnn_df.index,gnn_df["policy_rolling_val"],color="C1",alpha=1,linestyle="--",dashes=(5, 3))
    plt.ylim(min(cnn_df["policy_acc/train"].min(),gnn_df["policy_acc/train"].min(),cnn_df["policy_acc/val"].min(),gnn_df["policy_acc/val"].min()),
             max(cnn_df["policy_acc/train"].max(),gnn_df["policy_acc/train"].max(),cnn_df["policy_acc/val"].max(),gnn_df["policy_acc/val"].max()))
    plt.xlabel("epoch")
    plt.ylabel("policy accuracy")
    plt.plot([],[],color="black",linestyle="-",label="Train")
    plt.plot([],[],color="black",linestyle="--",label="Validation")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(os.path.join(img_path,"supervised_compare","policy_acc.svg"))
    plt.clf()

    plt.plot(gao_df.index,gao_df["value_acc_sign/train"],color="C2",alpha=0.2)
    plt.plot(gao_df.index,gao_df["value_rolling_train"],color="C2",alpha=1,label="Gao")
    plt.plot(cnn_df.index,cnn_df["value_acc_sign/train"],color="C0",alpha=0.2)
    plt.plot(gnn_df.index,gnn_df["value_acc_sign/train"],color="C1",alpha=0.2)
    plt.plot(cnn_df.index,cnn_df["value_rolling_train"],color="C0",alpha=1,label="U-Net")
    plt.plot(gnn_df.index,gnn_df["value_rolling_train"],color="C1",alpha=1,label="GNN")
    plt.ylim(min(gao_df["value_acc_sign/train"].min(),cnn_df["value_acc_sign/train"].min(),gnn_df["value_acc_sign/train"].min(),cnn_df["value_acc_sign/val"].min(),gnn_df["value_acc_sign/val"].min()),
             max(gao_df["value_acc_sign/train"].max(),cnn_df["value_acc_sign/train"].max(),gnn_df["value_acc_sign/train"].max(),cnn_df["value_acc_sign/val"].max(),gnn_df["value_acc_sign/val"].max()))
    # plt.xlabel("epoch")
    # plt.ylabel("training value sign accuracy")
    # plt.legend()
    # plt.savefig(os.path.join(img_path,"supervised_compare","value_acc_train.svg"))
    # plt.clf()

    plt.plot(gao_df.index,gao_df["value_acc_sign/val"],color="C2",alpha=0.2,linestyle="--",dashes=(5, 3))
    plt.plot(gao_df.index,gao_df["value_rolling_val"],color="C2",alpha=1,linestyle="--",dashes=(5, 3))
    plt.plot(cnn_df.index,cnn_df["value_acc_sign/val"],color="C0",alpha=0.2,linestyle="--",dashes=(5, 3))
    plt.plot(gnn_df.index,gnn_df["value_acc_sign/val"],color="C1",alpha=0.2,linestyle="--",dashes=(5, 3))
    plt.plot(cnn_df.index,cnn_df["value_rolling_val"],color="C0",alpha=1,linestyle="--",dashes=(5, 3))
    plt.plot(gnn_df.index,gnn_df["value_rolling_val"],color="C1",alpha=1,linestyle="--",dashes=(5, 3))
    plt.xlabel("epoch")
    plt.ylabel("value sign accuracy")
    plt.ylim(min(gao_df["value_acc_sign/train"].min(),cnn_df["value_acc_sign/train"].min(),gnn_df["value_acc_sign/train"].min(),cnn_df["value_acc_sign/val"].min(),gnn_df["value_acc_sign/val"].min()),
             max(gao_df["value_acc_sign/train"].max(),cnn_df["value_acc_sign/train"].max(),gnn_df["value_acc_sign/train"].max(),cnn_df["value_acc_sign/val"].max(),gnn_df["value_acc_sign/val"].max()))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # handles.append(dashline)
    plt.plot([],[],color="black",linestyle="-",label="Train")
    plt.plot([],[],color="black",linestyle="--",label="Validation")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(os.path.join(img_path,"supervised_compare","value_acc.svg"))
    plt.clf()

    plt.plot(gao_df.index,gao_df["loss/train"],color="C2",alpha=0.2)
    print(gao_df["loss/train"])
    plt.plot(gao_df.index,gao_df["loss_rolling_train"],color="C2",alpha=1,label="Gao")
    plt.plot(cnn_df.index,cnn_df["loss/train"],color="C0",alpha=0.2)
    plt.plot(gnn_df.index,gnn_df["loss/train"],color="C1",alpha=0.2)
    plt.plot(cnn_df.index,cnn_df["loss_rolling_train"],color="C0",alpha=1,label="U-Net")
    plt.plot(gnn_df.index,gnn_df["loss_rolling_train"],color="C1",alpha=1,label="GNN")
    plt.ylim(min(gao_df["loss/train"].min(),cnn_df["loss/train"].min(),gnn_df["loss/train"].min(),cnn_df["loss/val"].min(),gnn_df["loss/val"].min()),
             max(gao_df["loss/train"].max(),cnn_df["loss/train"].max(),gnn_df["loss/train"].max(),cnn_df["loss/val"].max(),gnn_df["loss/val"].max()))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    plt.savefig(os.path.join(img_path,"supervised_compare","loss_train.svg"))
    plt.clf()

    plt.plot(gao_df.index,gao_df["loss/val"],color="C2",alpha=0.2)
    plt.plot(gao_df.index,gao_df["loss_rolling_val"],color="C2",alpha=1,label="Gao")
    plt.plot(cnn_df.index,cnn_df["loss/val"],color="C0",alpha=0.2)
    plt.plot(gnn_df.index,gnn_df["loss/val"],color="C1",alpha=0.2)
    plt.plot(cnn_df.index,cnn_df["loss_rolling_val"],color="C0",alpha=1,label="U-Net")
    plt.plot(gnn_df.index,gnn_df["loss_rolling_val"],color="C1",alpha=1,label="GNN")
    plt.ylim(min(gao_df["loss/train"].min(),cnn_df["loss/train"].min(),gnn_df["loss/train"].min(),cnn_df["loss/val"].min(),gnn_df["loss/val"].min()),
             max(gao_df["loss/train"].max(),cnn_df["loss/train"].max(),gnn_df["loss/train"].max(),cnn_df["loss/val"].max(),gnn_df["loss/val"].max()))
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.legend()
    plt.savefig(os.path.join(img_path,"supervised_compare","loss_val.svg"))
    plt.clf()


def curriculum_stuff():
    icy_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"icy_resonance.csv"))
    beaming_df:pd.DataFrame = pd.read_csv(os.path.join(csv_path,"beaming_firecracker.csv"))
    combo_df = pd.concat([icy_df,beaming_df])
    combo_df = combo_df.sort_values("game_frame")
    plt.plot(combo_df["game_frame"]/1000000,combo_df["hidden_channels"])
    plt.xlabel("game frame (millions)")
    plt.ylabel("hidden channels")
    plt.savefig(os.path.join(img_path,"hidden_channels.svg"))
    plt.cla()
    plt.plot(combo_df["game_frame"]/1000000,combo_df["num_layers"])
    plt.xlabel("game frame (millions)")
    plt.ylabel("number of layers")
    plt.savefig(os.path.join(img_path,"num_layers.svg"))
    plt.cla()
    plt.plot(combo_df["game_frame"]/1000000,combo_df["hex_size"])
    plt.xlabel("game frame (millions)")
    plt.ylabel("Hex size")
    plt.savefig(os.path.join(img_path,"hex_size.svg"))
    plt.cla()


def get_run_df(name="project.csv",run_name="icv7ozhh",project_name="rainbow_hex"):
    api = wandb.Api()
    run = api.run(f"yannikkellerde/{project_name}/{run_name}")
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
    # get_run_df(name="icy_resonance.csv",run_name="40lz3z29")
    # get_run_df(name="beaming_firecracker.csv",run_name="rc0hl84y")
    # get_run_df(name="supervised_cnn.csv",run_name="87i7wr1b",project_name="HexAra")
    # get_run_df(name="supervised_gnn.csv",run_name="2moj3q5f",project_name="HexAra")
    # get_run_df(name="supervised_gao.csv",run_name="8pz9wfj3",project_name="HexAra")
    supervised_stuff()
    # curriculum_stuff()
    # get_big_df()
    # vs_random_winrate()
    # losses()
    # runtime_game_frame()
