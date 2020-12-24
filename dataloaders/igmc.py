from .base import AbstractDataloader
from .utils import *
from config import DATASET_ROOT_FOLDER

import torch
from torch_geometric.data import Dataset
from scipy import sparse
import numpy as np
import time


class IGMCDataloader(AbstractDataloader):
    @classmethod
    def code(cls):
        return 'igmc'

    def _get_dataset(self, mode):
        if mode == 'train':
            return self._get_train_dataset()
        elif mode == 'val':
            return self._get_eval_dataset('val')
        else:
            return self._get_eval_dataset('test')

    def _get_train_dataset(self):
        dataset = IGMCDataSet(args=self.args, matrix=self.adj_matrix, sp_matrix=self.train_sparse)
        return dataset

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = IGMCDataSet(args=self.args, matrix=self.adj_matrix, sp_matrix=self.val_sparse)
        else:
            dataset = IGMCDataSet(args=self.args, matrix=self.adj_matrix, sp_matrix=self.test_sparse)
        return dataset


class IGMCDataSet(Dataset):
    def __init__(self, args, matrix, sp_matrix):
        super(IGMCDataSet, self).__init__(DATASET_ROOT_FOLDER)
        self.args = args
        self.matrix = matrix
        self.sp_matrix = sp_matrix
        self.SRI = SparseRowIndexer(matrix)
        self.SCI = SparseColIndexer(matrix.tocsc())

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        return

    def process(self):
        return

    def __len__(self):
        return self.sp_matrix.shape[0]

    def get(self, index):
        # start = time.time()
        u, v = self.sp_matrix[index, 0], self.sp_matrix[index, 1]
        y = self.sp_matrix[index, 2]
        # print(u, v)
        data = extract_subgraph(self.SRI, self.SCI, u, v, y, max_nodes=self.args.max_nodes)
        # end = time.time()
        # print("get item time taken: ", end-start)
        return data


# The below classes are directly from the official IGMC source code.
class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return sp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return sp.csc_matrix((data, indices, indptr), shape=shape)
