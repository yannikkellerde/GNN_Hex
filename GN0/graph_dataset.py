import torch
from torch_geometric.data import InMemoryDataset, download_url
import pickle
from GN0.generate_training_data import generate_graphs

class SupervisedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SupervisedDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ['data_list.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        graphs = generate_graphs(2)
        with open(self.raw_file_names[0], 'wb') as f:
            pickle.dump(graphs, f)

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_file_names[0], 'rb') as f:
            graphs = pickle.load(f)

        if self.pre_filter is not None:
            graphs = [data for data in graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            graphs = [self.pre_transform(data) for data in graphs]

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])