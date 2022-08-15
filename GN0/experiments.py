from torch_geometric.nn.models import GCN, GraphSAGE
from torch_geometric.nn.norm import GraphNorm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from GN0.graph_dataset import SupervisedDataset,winpattern_pre_transform,hex_pre_transform
from GN0.train_GCN import train_gcn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

model_types = [GCN,GraphSAGE]
model_map = {x.__name__:x for x in model_types}
norm_types = [GraphNorm]
norm_map = {x.__name__:x for x in norm_types}

default_dataset_hyperparams = dict(root='./hexdata',game_type="hex",num_graphs=5000,drop=True)
default_model_hyperparams = dict(in_channels=3,out_channels=1,num_layers=13,hidden_channels=32,norm="GraphNorm",act="relu")
default_train_hyperparams = dict(lr=0.002,batch_size=256,model_name="GCN_hex",epochs=200)
default_all_hyperparams = {**default_dataset_hyperparams,**default_model_hyperparams,**default_train_hyperparams,"logdir_name":"run/GNNs_master","model_type":"GraphSAGE"} 

def run_single_experiment(all_hyperparams):
    dataset_hyperparams = {key:all_hyperparams[key] for key in default_dataset_hyperparams.keys()}
    model_hyperparams = {key:all_hyperparams[key] for key in default_model_hyperparams.keys()}
    train_hyperparams = {key:all_hyperparams[key] for key in default_train_hyperparams.keys()}

    writer = SummaryWriter(all_hyperparams["logdir_name"])
    dataset = SupervisedDataset(device=device, pre_transform=hex_pre_transform, **dataset_hyperparams)
    model_hyperparams["norm"] = norm_map[model_hyperparams["norm"]](model_hyperparams["hidden_channels"])
    model = model_map[all_hyperparams["model_type"]](**model_hyperparams).to(device)
    ev,loss = train_gcn(model,dataset,loss_func=F.mse_loss,eval_func=F.mse_loss,writer=writer,hparams_to_log=all_hyperparams,**train_hyperparams)

def run_experiments():
    hyper_param_sets = [
        {"model_type":"GraphSAGE",
         "lr":0.001,
         "model_name":"GraphSAGE_hex",
         "logdir_name":"run/GraphSAGE_hex"},
        {"model_type":"GCN",
         "lr":0.001,
         "model_name":"GCN_hex",
         "logdir_name":"run/GCN_hex"}
    ]
    for hyper_params in hyper_param_sets:
        params = default_all_hyperparams.copy()
        params.update(hyper_params)
        run_single_experiment(params)

if __name__ == "__main__":
    run_experiments()
