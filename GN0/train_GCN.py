from GN0.GCN import GCN
from GN0.generate_training_data import generate_graphs
from GN0.graph_dataset import SupervisedDataset,pre_transform
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torchmetrics import Accuracy
from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

dataset = SupervisedDataset(root='./data/', device=device, pre_transform=pre_transform)

model = GCN(3,2,conv_layers=8,conv_dim=64).to(device)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

acc_accumulate = Accuracy().to(device)

def eval(loader):
    losses = []
    accuracies = []
    for batch in loader:
        out = model(batch)
        loss = F.binary_cross_entropy(out[batch.test_mask], batch.y[batch.test_mask])
        accuracy = acc_accumulate(out[batch.test_mask].flatten(), batch.y[batch.test_mask].flatten().long())
        accuracies.append(accuracy)
        losses.append(loss)
    print("Testing accuracy:", sum(accuracies) / len(accuracies))
    print("Testing loss:", sum(losses) / len(losses))

model.train()
for _ in trange(2000):
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.binary_cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss)
    print("training loss:",sum(losses) / len(losses))
    eval(loader)

torch.save(model.state_dict(), './model/GCN_model.pt')