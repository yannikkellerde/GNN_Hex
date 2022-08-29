from torch_geometric.nn.models import GCN, GraphSAGE, GAT, PNA
from datetime import datetime
from torch_geometric.nn.norm import GraphNorm
import torch
import torch.nn.functional as F
from GN0.util import SummaryWriter
from GN0.graph_dataset import SupervisedDataset,winpattern_pre_transform,hex_pre_transform
from GN0.train_GCN import train_gcn
from GN0.models import cachify_gnn,PolicyValueGNN,CachedGraphNorm
from torch_geometric.nn import SAGEConv,GCNConv,GATv2Conv,PNAConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

model_types = [GCN,GraphSAGE,GAT,PNA]
model_map = {x.__name__:x for x in model_types}
norm_types = [CachedGraphNorm]
norm_map = {x.__name__:x for x in norm_types}
policy_heads = [SAGEConv,GCNConv,GATv2Conv,PNAConv]
policy_map = {x.__name__:x for x in policy_heads}

dataset_params = ["root","game_type","num_graphs","drop"]
model_params = ["in_channels","num_layers","hidden_channels","norm","act","policy_head","project","aggr"]
train_params = ["lr","batch_size","model_name","epochs","policy_weighting"]
default_dataset_hyperparams = dict(root='./data/policy_value',game_type="hex",num_graphs=5000,drop=True)
default_model_hyperparams = dict(in_channels=4,num_layers=13,hidden_channels=32,norm="CachedGraphNorm",act="relu",policy_head="SAGEConv")
default_train_hyperparams = dict(lr=0.002,batch_size=256,model_name="GCN_hex",epochs=200,policy_weighting=0.01)
default_all_hyperparams = {**default_dataset_hyperparams,**default_model_hyperparams,**default_train_hyperparams,"logdir_name":"run/GNNs_master","model_type":"GraphSAGE"} 

def run_single_experiment(all_hyperparams):
    dataset_hyperparams = {key:all_hyperparams[key] for key in dataset_params if key in all_hyperparams}
    model_hyperparams = {key:all_hyperparams[key] for key in model_params if key in all_hyperparams}
    train_hyperparams = {key:all_hyperparams[key] for key in train_params if key in all_hyperparams}

    dataset = SupervisedDataset(device=device, pre_transform=hex_pre_transform, **dataset_hyperparams)
    if model_hyperparams["norm"] is not None:
        model_hyperparams["norm"] = norm_map[model_hyperparams["norm"]](model_hyperparams["hidden_channels"])
    model_hyperparams["policy_head"] = policy_map[model_hyperparams["policy_head"]]
    if hasattr(model_hyperparams["norm"],"supports_cache") and model_hyperparams["norm"].supports_cache:
        body_model = cachify_gnn(model_map[all_hyperparams["model_type"]]) 
    else:
        body_model = model_map[all_hyperparams["model_type"]]
    model = PolicyValueGNN(body_model,**model_hyperparams).to(device)
    all_hyperparams["total_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer = SummaryWriter(all_hyperparams["logdir_name"])
    metrics = {'Test/loss': None, 'Train/loss': None, 'Train/value_loss':None, 'Train/policy_loss':None, 'Test/value_loss':None,'Test/policy_loss':None}
    ev,loss = train_gcn(model,dataset,writer=writer,hparams_to_log=all_hyperparams,**train_hyperparams)
    writer.add_hparams(all_hyperparams,metrics)
    writer.close()

def run_experiments():
    hyper_param_sets = [
        {"model_type":"GraphSAGE",
         "lr":0.001,
         "model_name":"GraphSAGE_max_aggr",
         "policy_head":"SAGEConv",
         "logdir_name":"run/GraphSAGE_max_aggr",
         "hidden_channels":32,
         "aggr":"max"},
        {"model_type":"GraphSAGE",
         "lr":0.002,
         "model_name":"GraphSAGE_higher_lr",
         "policy_head":"SAGEConv",
         "logdir_name":"run/GraphSAGE_higher_lr"},
        {"model_type":"GraphSAGE",
         "lr":0.001,
         "model_name":"GraphSAGE_thick_deep",
         "policy_head":"SAGEConv",
         "logdir_name":"run/GraphSAGE_thick_deep",
         "num_layers":18,
         "hidden_channels":48},
    ]
    for hyper_params in hyper_param_sets:
        hyper_params["logdir_name"] = f'{hyper_params["logdir_name"]}_{datetime.now().isoformat()}'
        params = default_all_hyperparams.copy()
        params.update(hyper_params)
        run_single_experiment(params)

if __name__ == "__main__":
    run_experiments()
