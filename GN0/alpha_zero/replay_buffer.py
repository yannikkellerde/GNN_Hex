import torch
from torch_geometric.data import Data,Batch
from torch_geometric.loader.dataloader import DataLoader

class ReplayBuffer:
    def __init__(self, burnin, capacity, device):
        self.capacity = capacity
        self.burnin = burnin
        self.buffer = []
        self.nextwrite = 0
        self.device = device

    def put(self,state,pi,v):
        state.pi = torch.FloatTensor(pi).to(self.device)
        state.v = torch.FloatTensor([v]).to(self.device)
        state.to(self.device)
        if len(self.buffer) < self.capacity:
            self.buffer.append(state)
        else:
            self.buffer[self.nextwrite % self.capacity] = state
            self.nextwrite += 1
    
    def get_data_loader(self, batch_size=128, shuffle=True):
        return DataLoader(self.buffer,batch_size=batch_size,shuffle=shuffle)

    @property
    def burnedin(self):
        return len(self) >= self.burnin

    def __len__(self):
        return len(self.buffer)

