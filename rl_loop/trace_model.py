import torch
from GN0.torch_script_models import PNA_torch_script,SAGE_torch_script, get_current_model
import time
import os

if __name__ == "__main__":
    model=get_current_model()
    traced = torch.jit.script(model)
    path = "data/RL/model/HexAra/torch_script_model.pt"
    os.makedirs(os.path.dirname(path),exist_ok=True)

    traced.save(path)
    # traced.save("data/test_model.pt")

    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':torch.optim.SGD(model.parameters(),0.1).state_dict(),
        },"data/RL/model/HexAra/weights_torch_script.pt")

    num_graphs = 200
    num_nodes = 100

    device = "cuda"
    x = torch.zeros([num_graphs*num_nodes,3]).to(device)
    edge_index = torch.stack([torch.cat([torch.cat((torch.arange(i+1,i+num_nodes-1),torch.arange(i,i+num_nodes-2))) for i in range(0,num_nodes*num_graphs,num_nodes)]),torch.cat([torch.cat((torch.arange(i,i+num_nodes-2),torch.arange(i+1,i+num_nodes-1))) for i in range(0,num_nodes*num_graphs,num_nodes)])]).long().to(device);
    batch_ptr = torch.arange(0,num_nodes*num_graphs+1,num_nodes).long().to(device);
    graph_indices = torch.cat([torch.ones(num_nodes)*i for i in range(num_graphs)]).long().to(device);
    traced = traced.to(device)
    start = time.time()
    for i in range(50):
        output = traced(x,edge_index,graph_indices,batch_ptr)
    print(output)
    print(time.time()-start)

