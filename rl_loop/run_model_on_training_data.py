from rl_loop.dataset_loader import load_pgn_dataset
import torch
import os

dataset = load_pgn_dataset(q_value_ratio=0.15)
path = "../data/RL/model/HexAra/torch_script_model.pt"
weights = "../data/RL/model/HexAra/weights-0.07591-475.77017-0.984-0.677-0054.pt"
os.makedirs(os.path.dirname(path),exist_ok=True)


model = torch.jit.load(path)
device = "cuda"
# checkpoint = torch.load(weights)
# model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
data = dataset[0].to(device)
ptr = torch.tensor([0,len(data.x)],dtype=torch.int64).to(device)
batch = torch.zeros(data.x.shape[0],dtype=torch.int64).to(device)
print(batch.shape,data.x.shape)
output = model(data.x,data.edge_index,batch,ptr)
for i in torch.exp(output[0]):
    print(i)
