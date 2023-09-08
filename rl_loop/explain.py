from dataset_loader import get_loader
from collections import namedtuple
from GN0.torch_script_models import get_current_model 
import torch
import torch.nn.functional as F

device="cpu"

def plot_explaination(model,data,optimizer):
    data = data.to(device)
    optimizer.zero_grad()
    input = torch.autograd.Variable(data.x,requires_grad=True)
    out = model(input,data.edge_index,data.batch,data.ptr)
    # loss = F.binary_cross_entropy(out, data.y)
    loss = F.mse_loss(out[1], data.y.float())
    loss.backward()
    input_grads = input.grad
    saliency_map_weights = saliency_map(input_grads)
    print(saliency_map_weights.shape)


def saliency_map(input_grads):
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    return torch.tensor(node_saliency_map)


if __name__=="__main__":
    TrainConfig = namedtuple("TrainConfig","q_value_ratio batch_size cpu_count")
    tc = TrainConfig(q_value_ratio=0,batch_size=1,cpu_count=1)
    loader = get_loader(train_config=tc,dataset_type="train")
    model = get_current_model(net_type="SAGE")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    for data in loader:
        plot_explaination(model,data,optimizer)
