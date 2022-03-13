from GN0.GCN import GCN, perfs
from GN0.generate_training_data import generate_graphs
from GN0.graph_dataset import SupervisedDataset,pre_transform
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
from tqdm import trange,tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

dataset = SupervisedDataset(root='./data/', device=device, pre_transform=pre_transform)

model = GCN(3,2,conv_layers=8,conv_dim=16,global_dim=16).to(device)
#loader = DataLoader(dataset, batch_size=64, shuffle=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

acc_accumulate = Accuracy().to(device)

def eval(loader):
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch in tqdm(loader):
            out = model(batch)
            loss = F.binary_cross_entropy(out[batch.test_mask], batch.y[batch.test_mask])
            accuracy = acc_accumulate(out[batch.test_mask].flatten(), batch.y[batch.test_mask].flatten().long())
            accuracies.append(accuracy)
            losses.append(loss)
    print("Testing accuracy:", sum(accuracies) / len(accuracies))
    print("Testing loss:", sum(losses) / len(losses))
    return sum(accuracies) / len(accuracies)

model.train()
best_acc = 0
for _ in trange(200):
    losses = []
    for batch in tqdm(loader):
        optimizer.zero_grad()
        out = model(batch)
        loss = F.binary_cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss)

    #print({key:sum(value)/len(value) for key,value in perfs.items()})
    print("training loss:",sum(losses) / len(losses))
    acc = eval(loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model/GCN_model.pt")

torch.save(model.state_dict(), './model/GCN_final_model.pt')