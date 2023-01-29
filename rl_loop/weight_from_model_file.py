import torch
from GN0.torch_script_models import PNA_torch_script,SAGE_torch_script, get_current_model
import sys

if __name__ == "__main__":
    model = torch.jit.load(sys.argv[1])
    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':torch.optim.SGD(model.parameters(),0.1).state_dict()
    },sys.argv[2])
