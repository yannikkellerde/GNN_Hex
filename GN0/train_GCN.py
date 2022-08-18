from GN0.models import GCN_with_glob, perfs
import numpy as np
import os
from GN0.graph_dataset import SupervisedDataset,winpattern_pre_transform,hex_pre_transform
from GN0.generate_training_data import generate_hex_graphs, generate_winpattern_game_graphs
from GN0.convert_graph import convert_node_switching_game, convert_winpattern_game
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
from torch_geometric.nn import GCNConv
from tqdm import trange,tqdm
from typing import Callable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

#loader = DataLoader(dataset, batch_size=64, shuffle=True)
def train_gcn(model:torch.nn.Module,dataset,model_name:str,loss_func:Callable,eval_func:Callable,writer:SummaryWriter,
              categorical_eval=False,maximizing_eval=False,epochs=2000,lr=0.002,batch_size=256,hparams_to_log=None):
    def save_model(name_modifier):
        save_dic = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ev': ev,
            'hparams':hparams_to_log,
            }
        if hasattr(model,"supports_cache") and model.supports_cache:
            save_dic["cache"] = model.export_norm_cache()
        torch.save(save_dic, os.path.join("model",f"{model_name}_{name_modifier}.pt"))
    eval_func_name = eval_func.__name__ if hasattr(eval_func,"__name__") else eval_func.__class__.__name__
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    writer.add_graph(model,(batch.x,batch.edge_index))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    best_ev = np.inf
    best_loss = np.inf
    for epoch in trange(epochs):
        losses = []
        eval_metrics = []
        for i,batch in tqdm(enumerate(loader)):
            optimizer.zero_grad()
            if i==len(loader)-1 and hasattr(model,"supports_cache") and model.supports_cache:
                policy,value = model(batch.x,batch.edge_index,batch.batch,set_cache=True)
            else:
                policy,value = model(batch.x,batch.edge_index,batch.batch)
            loss = loss_func(policy[batch.train_mask], batch.y[batch.train_mask])
            if categorical_eval:
                eval_res = eval_func(out[batch.test_mask].flatten(), batch.y[batch.test_mask].flatten().long())
            else:
                eval_res = eval_func(out[batch.test_mask], batch.y[batch.test_mask])
            eval_metrics.append(eval_res)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        lo = sum(losses)/len(losses)
        ev = sum(eval_metrics)/len(eval_metrics)
        if lo<best_loss:
            best_loss=lo
        writer.add_scalar("Loss/train",lo,epoch)
        writer.add_scalar("Test/"+eval_func_name,ev,epoch)



        #print({key:sum(value)/len(value) for key,value in perfs.items()})
        print("Epoch",epoch,"training loss:",sum(losses) / len(losses))
        print("Testing evaluation:",ev)
        
        if (maximizing_eval and -ev<best_ev) or (not maximizing_eval and ev < best_ev):
            best_ev = ev
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
