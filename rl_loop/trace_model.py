import torch
from GN0.models import get_pre_defined

if __name__ == "__main__":
    model = get_pre_defined("HexAra")
    traced = torch.jit.script(model)
    traced.save("../data/RL/model/HexAra/graph_sage_model.pt")

    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':torch.optim.SGD(model.parameters(),0.1).state_dict(),
        },"../data/RL/model/HexAra/weights_graph_sage.pt")

    x = torch.ones([20,3])
    edge_index = torch.tensor([[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
    graph_indices = torch.tensor(([0]*10)+([1]*10))
    model(x,edge_index,graph_indices)

