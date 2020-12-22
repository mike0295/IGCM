from config import RAW_DATASET_ROOT_FOLDER
from .utils import *

import random
from abc import *
from pathlib import Path
import tempfile
import shutil
import pickle
import pandas as pd
import os

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        self.preprocessing()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def load_data(self, df, seed=1234):
        data_array = df.values.tolist()
        # Randomly shuffle data
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)
        u_map = data_array[:, 0]
        v_map = data_array[:, 1]
        u_map, u_dict, num_users = map_data(u_map)
        v_map, v_dict, num_items = map_data(v_map)
        ratings = data_array[:, 2]

        return u_map, u_dict, num_users, v_map, v_dict, num_items, ratings

    def preprocessing(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        self.download_raw_dataset()
        df = self.load_ratings_df()
        u_map, u_dict, num_users, v_map, v_dict, num_items, ratings = self.load_data(df)

        train_matrix, val_matrix, test_matrix = trainvaltest_split(ratings, u_map, v_map, num_users, num_items)

        dataset = {
                    'train_matrix': train_matrix,
                    'val_matrix': val_matrix,
                    'test_matrix': test_matrix
        }

        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and \
                all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_preprocessed'.format(self.code())
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')