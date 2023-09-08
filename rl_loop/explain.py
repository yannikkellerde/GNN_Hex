from dataset_loader import get_loader
from GN0.torch_script_models import get_current_model 
import torch
import torch.nn.functional as F

device="cpu"

def plot_explaination(model,data,optimizer):
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data)
    # loss = F.binary_cross_entropy(out, data.y)
    loss = F.mse_loss(out, data.y)
    loss.backward()
    input_grads = model.input.grad.view(40, 8)
    saliency_map_weights = saliency_map(input_grads)
    print(saliency_map_weights)


def saliency_map(input_grads):
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    return node_saliency_map


if __name__=="__main__":
    loader = get_loader()
