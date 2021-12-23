from GN0.GCN import GCN
from GN0.generate_training_data import generate_graphs
from GN0.graph_dataset import SupervisedDataset,pre_transform
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

dataset = SupervisedDataset(root='./data/', device=device, pre_transform=pre_transform)

print(dataset[0],len(dataset))

model = GCN(3,2,conv_layers=4,conv_dim=64).to(device)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

model.train()
for _ in trange(2000):
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.binary_cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    print(sum(losses) / len(losses))