from abc import *
from torch_geometric.data import DataLoader
import numpy as np
import scipy.sparse as sp


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.save_folder = dataset._get_preprocessed_folder_path()
        self.dataset = dataset.load_dataset()
        self.adj_matrix = self.dataset['adj_matrix']
        self.train = self.dataset['train_matrix']
        self.val = self.dataset['val_matrix']
        self.test = self.dataset['test_matrix']
        self.train_sparse = np.array(sp.find(self.train)).transpose()
        self.val_sparse = np.array(sp.find(self.val)).transpose()
        self.test_sparse = np.array(sp.find(self.test)).transpose()

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def get_pytorch_dataloaders(self):
        train_loaders = self._get_dataloaders('train')
        val_loaders = self._get_dataloaders('val')
        test_loaders = self._get_dataloaders('test')
        return train_loaders, val_loaders, test_loaders

    def _get_dataloaders(self, mode):
        batch_size = {'train': self.args.train_batch_size,
                      'val': self.args.val_batch_size,
                      'test': self.args.test_batch_size}[mode]

        dataset = self._get_dataset(mode)
        # print(dataset)
        dataloader = DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               )
        return dataloader

    @abstractmethod
    def _get_dataset(self, mode):
        pass