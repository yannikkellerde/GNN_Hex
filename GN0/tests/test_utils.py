from GN0.util import graph_cross_entropy
import torch_geometric.utils
import torch
from torch import nn

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def test_cross_entropy():
    targets = torch.tensor([1,2,3,4,5,6,1,2,3,4],dtype=torch.float32)
    pred = torch.tensor([1,2,3,4,5,6,1,2,3,4],dtype=torch.float32)
    index = torch.tensor([0,0,0,0,0,0,1,1,1,1])
    ptr = torch.tensor([0,6,10])
    normal_ce = cross_entropy
    targets = torch_geometric.utils.softmax(targets,ptr=ptr)
    normal1 = normal_ce(pred[:6].unsqueeze(0),targets[:6].unsqueeze(0))
    normal2 = normal_ce(pred[6:].unsqueeze(0),targets[6:].unsqueeze(0))
    loss1 = graph_cross_entropy(pred,targets,index=index)
    loss2 = graph_cross_entropy(pred,targets,ptr=ptr)
    print(normal1,normal2,torch.mean(torch.tensor([normal1,normal2])))
    print(loss1,loss2)


if __name__ == "__main__":
    test_cross_entropy()
