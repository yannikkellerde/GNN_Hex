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


default_dataset_hyperparams = dict(root='./data/policy_value',game_type="hex",num_graphs=5000,drop=True)
default_model_hyperparams = dict(in_channels=4,num_layers=13,hidden_channels=32,norm="CachedGraphNorm",act="relu",policy_head="SAGEConv")
default_train_hyperparams = dict(lr=0.002,batch_size=256,model_name="GCN_hex",epochs=200,policy_weighting=0.01)
default_all_hyperparams = {**default_dataset_hyperparams,**default_model_hyperparams,**default_train_hyperparams,"logdir_name":"run/GNNs_master","model_type":"GraphSAGE"} 

def run_single_experiment(all_hyperparams):
    dataset_hyperparams = {key:all_hyperparams[key] for key in default_dataset_hyperparams.keys()}
    model_hyperparams = {key:all_hyperparams[key] for key in default_model_hyperparams.keys()}
    train_hyperparams = {key:all_hyperparams[key] for key in default_train_hyperparams.keys()}

    writer = SummaryWriter(all_hyperparams["logdir_name"])
    metrics = {'Test/loss': None, 'Train/loss': None, 'Train/value_loss':None, 'Train/policy_loss':None, 'Test/value_loss':None,'Test/policy_loss':None}
    dataset = SupervisedDataset(device=device, pre_transform=hex_pre_transform, **dataset_hyperparams)
    if model_hyperparams["norm"] is not None:
        model_hyperparams["norm"] = norm_map[model_hyperparams["norm"]](model_hyperparams["hidden_channels"])
    model_hyperparams["policy_head"] = policy_map[model_hyperparams["policy_head"]]
    if hasattr(model_hyperparams["norm"],"supports_cache") and model_hyperparams["norm"].supports_cache:
        body_model = cachify_gnn(model_map[all_hyperparams["model_type"]]) 
    else:
        body_model = model_map[all_hyperparams["model_type"]]
    model = PolicyValueGNN(body_model,**model_hyperparams).to(device)
    ev,loss = train_gcn(model,dataset,writer=writer,hparams_to_log=all_hyperparams,**train_hyperparams)
    writer.add_hparams(all_hyperparams,metrics)
    writer.close()

def run_experiments():
    hyper_param_sets = [
        {"model_type":"GraphSAGE",
         "lr":0.001,
         "model_name":"GraphSAGE_deep_thin",
         "policy_head":"SAGEConv",
         "logdir_name":"run/GraphSAGE_deep_thin",
         "num_layers":18,
         "hidden_channels":16},
        {"model_type":"GraphSAGE",
         "lr":0.001,
         "model_name":"GraphSAGE_shallow_thick",
         "policy_head":"SAGEConv",
         "logdir_name":"run/GraphSAGE_shallow_thick",
         "num_layers":9,
         "hidden_channels":48},
        {"model_type":"GraphSAGE",
         "lr":0.001,
         "model_name":"GraphSAGE_no_norm",
         "policy_head":"SAGEConv",
         "logdir_name":"run/GraphSAGE_no_norm",
         "norm":None},
        {"model_type":"GAT",
         "lr":0.001,
         "model_name":"GATv2",
         "policy_head":"GATv2Conv",
         "logdir_name":"run/GATv2"},
        {"model_type":"PNA",
         "lr":0.001,
         "model_name":"PNA",
         "policy_head":"PNAConv",
         "logdir_name":"run/PNA"},
    ]
    for hyper_params in hyper_param_sets:
        hyper_params["logdir_name"] = f'{hyper_params["logdir_name"]}_{datetime.now().isoformat()}'
        params = default_all_hyperparams.copy()
        params.update(hyper_params)
        run_single_experiment(params)

if __name__ == "__main__":
    run_experiments()
