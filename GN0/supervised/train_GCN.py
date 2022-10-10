from GN0.models import GCN_with_glob, perfs
import numpy as np
import os
from GN0.supervised.graph_dataset import SupervisedDataset,winpattern_pre_transform,hex_pre_transform
from GN0.supervised.generate_training_data import generate_hex_graphs, generate_winpattern_game_graphs
from GN0.util.convert_graph import convert_node_switching_game, convert_winpattern_game
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
from torch_geometric.nn import GCNConv
from tqdm import trange,tqdm
from typing import Callable
from torch.nn import BCELoss,CrossEntropyLoss, MSELoss
from GN0.util.util import graph_cross_entropy
import torch_geometric.utils
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

#loader = DataLoader(dataset, batch_size=64, shuffle=True)
def train_gcn(model:torch.nn.Module,dataset,model_name:str,writer:SummaryWriter,
              epochs=2000,lr=0.002,batch_size=256,hparams_to_log=None,policy_weighting=1):
    def save_model(name_modifier):
        save_dic = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': loss_logs,
            'hparams':hparams_to_log,
            }
        if hasattr(model,"supports_cache") and model.supports_cache:
            save_dic["cache"] = model.export_norm_cache()
        torch.save(save_dic, os.path.join("model",f"{model_name}_{name_modifier}.pt"))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # batch = next(iter(train_loader))
    # writer.add_graph(model,(batch.x,batch.edge_index))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_loss_func = BCELoss()
    mse = MSELoss()
    # BCELoss() is correct for value
    # CrossEntropyLoss() is more correct, but does not work that easily, because we have one probabilty distribution per graph.

    model.train()
    best_ev = np.inf
    best_loss = np.inf
    for epoch in trange(epochs):
        loss_logs = defaultdict(list)
        for i,batch in tqdm(enumerate(train_loader),total=len(train_loader)):
            tt_mask = batch.train_mask | batch.test_mask
            optimizer.zero_grad()
            if i==len(train_loader)-1 and hasattr(model,"supports_cache") and model.supports_cache:
                policy,value = model(batch.x,batch.edge_index,graph_indices=batch.batch,set_cache=True)
            else:
                policy,value = model(batch.x,batch.edge_index,graph_indices=batch.batch)
            policy_loss = mse(policy[tt_mask],batch.y[tt_mask])
            # policy_cross_entropy = graph_cross_entropy(policy[tt_mask].squeeze(), torch_geometric.utils.softmax(batch.y[tt_mask],index=batch.batch[tt_mask]).squeeze(), index=batch.batch[tt_mask])
            # print(policy.max(),policy.sum(),policy.min(),batch.y.max(),batch.y.sum(),batch.y.min())
            # exit()
            value_loss = value_loss_func(value.squeeze(),batch.global_y)
            loss = policy_loss*policy_weighting+value_loss
            loss.backward()
            optimizer.step()
            loss_logs["Train/loss"].append(loss)
            loss_logs["Train/value_loss"].append(value_loss)
            loss_logs["Train/policy_loss"].append(policy_loss)
        with torch.no_grad():
            for batch in tqdm(test_loader):
                tt_mask = batch.train_mask | batch.test_mask
                policy,value = model(batch.x,batch.edge_index,graph_indices=batch.batch)
                policy_cross_entropy = graph_cross_entropy(policy[tt_mask].squeeze(), torch_geometric.utils.softmax(batch.y[tt_mask],index=batch.batch[tt_mask]).squeeze(), index=batch.batch[tt_mask])
                value_loss = value_loss_func(value.squeeze(),batch.global_y)
                policy_mse = mse(policy[tt_mask],batch.y[tt_mask])
                value_mse = mse(value.squeeze(),batch.global_y)
                loss = policy_mse*policy_weighting+value_loss
                loss_logs["Test/loss"].append(loss)
                loss_logs["Test/value_loss"].append(value_loss)
                loss_logs["Test/value_mse"].append(value_mse)
                loss_logs["Test/policy_loss"].append(policy_mse)
                loss_logs["Test/policy_cross_entropy"].append(policy_cross_entropy)

        loss_logs = {key:sum(value)/len(value) for key,value in loss_logs.items()}
        if loss_logs["Train/loss"]<best_loss:
            best_loss=loss_logs["Train/loss"]
        for key,value in loss_logs.items():
            writer.add_scalar(key,value,epoch)

        print("Epoch",epoch,loss_logs)
        
        if loss_logs["Test/loss"] < best_ev:
            best_ev = loss_logs["Test/loss"]
            save_model("best")
        elif epoch%100==0:
            save_model(str(epoch))
    save_model("final")
    return best_ev,best_loss

def train_hex():
    dataset_hyperparams = dict(root='./hexdata',game_type="hex",num_graphs=5000,drop=True)
    model_hyperparams = dict(conv_layers=13,conv_dim=32,parameter_sharing=False,instance_norm=True)
    train_hyperparams = dict(lr=0.002,batch_size=256,model_name="GCN_hex_unshared")
    all_hyperparams = {**dataset_hyperparams,**model_hyperparams,**train_hyperparams,"logdir_name":"run/GNNs_master2"} 
    writer = SummaryWriter(all_hyperparams["logdir_name"])
    writer.add_hparams(all_hyperparams,{"Train/loss":0,"Test/mse_loss":0})
    dataset = SupervisedDataset(device=device, pre_transform=hex_pre_transform, **dataset_hyperparams)
    model = GCN(3,1,**model_hyperparams).to(device)
    ev,loss = train_gcn(model,dataset,loss_func=F.mse_loss,eval_func=F.mse_loss,writer=writer,**train_hyperparams)


def train_qango():
    dataset = SupervisedDataset(root='./qangodata/', device=device, pre_transform=winpattern_pre_transform, game_type="qango", num_graphs=1000)
    model = GCN_with_glob(3,2,conv_layers=8,conv_dim=16,global_dim=16).to(device)
    train_gcn(model,dataset,"GCN_qango",loss_func=F.binary_cross_entropy,eval_func=Accuracy().to(device),categorical_eval=True,maximizing_eval=True)
if __name__=="__main__":
    train_hex()
