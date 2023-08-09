import torch
from GN0.torch_script_models import PNA_torch_script,SAGE_torch_script, get_current_model, Unet, Gao_baseline
from rl_loop.train_config import TrainConfig
import time
import os

if __name__ == "__main__":
    model=get_current_model(net_type=TrainConfig.net_type,hidden_channels=TrainConfig.hidden_channels,hidden_layers=TrainConfig.hidden_layers,policy_layers=TrainConfig.policy_layers,value_layers=TrainConfig.value_layers,in_channels=TrainConfig.in_channels,swap_allowed=TrainConfig.swap_allowed,norm=TrainConfig.norm)
    # model = Unet(3)
    # model = get_current_model("PV_CNN")
    # model = Gao_baseline(11,batch_norm=False)
    traced = torch.jit.script(model)
    path = "data/RL/model/HexAra/torch_script_model.pt"
    # path = "rl_loop/testmodel.pt"
    os.makedirs(os.path.dirname(path),exist_ok=True)

    traced.save(path)
    # traced.save("data/test_model.pt")

    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':torch.optim.SGD(model.parameters(),0.1).state_dict(),
        },"data/RL/model/HexAra/weights_torch_script.pt")

    # num_graphs = 200
    # num_nodes = 100

    # device = "cuda"
    # x = torch.zeros([num_graphs*num_nodes,3]).to(device)
    # edge_index = torch.stack([torch.cat([torch.cat((torch.arange(i+1,i+num_nodes-1),torch.arange(i,i+num_nodes-2))) for i in range(0,num_nodes*num_graphs,num_nodes)]),torch.cat([torch.cat((torch.arange(i,i+num_nodes-2),torch.arange(i+1,i+num_nodes-1))) for i in range(0,num_nodes*num_graphs,num_nodes)])]).long().to(device);
    # batch_ptr = torch.arange(0,num_nodes*num_graphs+1,num_nodes).long().to(device);
    # graph_indices = torch.cat([torch.ones(num_nodes)*i for i in range(num_graphs)]).long().to(device);
    # traced = traced.to(device)
    # start = time.time()
    # for i in range(50):
    #     output = traced(x,edge_index,graph_indices,batch_ptr)
    # print(output)
    # print(time.time()-start)

