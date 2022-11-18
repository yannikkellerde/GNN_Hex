import torch
from GN0.models import get_pre_defined

if __name__ == "__main__":
    model = get_pre_defined("HexAra")
    traced = torch.jit.script(model)
    traced.save("../data/RL/model/HexAra/graph_sage_model.pt")

    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':torch.optim.SGD(model.parameters(),0.1).state_dict(),
        },"../data/RL/model/HexAra/graph_sage_weights.pt")
