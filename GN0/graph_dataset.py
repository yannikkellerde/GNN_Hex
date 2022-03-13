# TODO: Bipartite Graph
import torch
from torch_geometric.data import InMemoryDataset, download_url
import pickle
from GN0.generate_training_data import generate_graphs_multiprocess
import numpy as np
import os

class SupervisedDataset(InMemoryDataset):
    num_data_creation_processes = 15

    def __init__(self, root, device="cpu", transform=None, pre_transform=None, num_graphs=10000):
        self.num_graphs = num_graphs
        super(SupervisedDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data = self.data.to(device)

    @property
    def raw_file_names(self):
        return [f'data_list_{i}.pkl' for i in range(SupervisedDataset.num_data_creation_processes)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        generate_graphs_multiprocess(self.num_graphs,self.raw_paths)
        print("graphs are generated")

    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        for path in self.raw_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    some_graphs = pickle.load(f)
                graphs.extend(some_graphs)

        if self.pre_filter is not None:
            graphs = [data for data in graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(data) for data in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

def pre_transform(data):
    train_mask = np.random.binomial(1, 0.8, len(data.y)).astype(bool)
    test_mask = ~train_mask
    data.train_mask = np.logical_and(train_mask, ~data.x[:, 0]).bool()
    data.test_mask = np.logical_and(test_mask, ~data.x[:, 0]).bool()
    data.x = data.x.float()
    data.y = data.y.float()
    return data