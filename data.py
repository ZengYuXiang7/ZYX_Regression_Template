# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch

from utils.dataloader import get_dataloaders


class experiment:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target.reshape(-1, 1)
        tensor = np.concatenate((X, y), axis=1)
        return tensor

    @staticmethod
    def preprocess_data(data, args):
        return data

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = self.get_train_valid_test_dataset(self.data, args)
        try:
            self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor, self.test_tensor, exper_type, args)
            self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
            args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')
        except:
            args.log.only_print(f'Executing machine learning')

    def get_dataset(self, train_tensor, valid_tensor, test_tensor, exper_type, args):
        return (
            TensorDataset(train_tensor, exper_type, args),
            TensorDataset(valid_tensor, exper_type, args),
            TensorDataset(test_tensor, exper_type, args)
        )

    def get_train_valid_test_dataset(self, tensor, args):
        np.random.shuffle(tensor)

        X = tensor[:, :-1]
        Y = tensor[:, -1].reshape(-1, 1)
        max_value = Y.max()
        # max_value = 1
        Y /= max_value

        train_size = int(len(tensor) * 0.3)
        valid_size = int(len(tensor) * 0.05)

        X_train = X[:train_size]
        Y_train = Y[:train_size]

        X_valid = X[train_size:train_size + valid_size]
        Y_valid = Y[train_size:train_size + valid_size]

        X_test = X[train_size + valid_size:]
        Y_test = Y[train_size + valid_size:]

        train_tensor = np.hstack((X_train, Y_train))
        valid_tensor = np.hstack((X_valid, Y_valid))
        test_tensor = np.hstack((X_test, Y_test))

        return train_tensor, valid_tensor, test_tensor, max_value


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor, exper_type, args):
        self.args = args
        self.tensor = tensor
        self.indices = exper_type.get_pytorch_index(tensor)

    def __getitem__(self, idx):
        output = self.indices[idx]
        inputs, value = torch.as_tensor(output[:-1]), torch.as_tensor(output[-1])
        inputs = inputs.to(torch.float)
        value = value.to(torch.float)
        return inputs, value

    def __len__(self):
        return self.indices.shape[0]



