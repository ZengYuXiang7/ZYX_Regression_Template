# coding : utf-8
# Author : yuxiang Zeng
import collections
import time
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import *
import torch
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.utils import set_settings, set_seed
from data import get_train_valid_test_dataset
global log
torch.set_default_dtype(torch.double)



class experiment:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        df = pd.read_excel('./datasets/数据预处理.xlsx')
        df = df.drop(['实验标号'], axis=1).values
        tensor = np.array(df)
        return tensor

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        return data

class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = get_train_valid_test_dataset(self.data, args)
        args.log.only_print(f'Train_length : {len(self.train_tensor)} Valid_length : {len(self.valid_tensor)} Test_length : {len(self.test_tensor)}')



class Model(torch.torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.log = args.log
    def forward(self, adjacency, features):
        pass

    def set_runid(self, runid):
        self.runid = runid
        self.log('-' * 80)
        self.log(f'Runid : {self.runid + 1}')

    def machine_learning_model_train_evaluation(self, train_x, train_y, valid_x, valid_y, test_x, test_y, max_value):
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import ParameterGrid
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        # print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'SVR': SVR(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
        }
        param_grids = {
            'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 9, 11, 15]},
            'SVR': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']},
            'DecisionTreeRegressor': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForestRegressor': {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 1],
                'max_depth': [3, 5, 7]
            }
        }
        results_dict = {}
        for name, model in models.items():
            self.log(f"模型: {name}")
            if name in param_grids:
                best_score = float('inf')
                best_params = None
                for params in ParameterGrid(param_grids[name]):
                    model.set_params(**params)
                    model.fit(train_x, train_y)
                    predictions = model.predict(valid_x)
                    score = mean_squared_error(valid_y, predictions)
                    if score < best_score:
                        best_score = score
                        best_params = params
                # print(f"{name} 最佳参数: {best_params}")
                model.set_params(**best_params)
                model.fit(train_x, train_y)
            else:
                model.fit(train_x, train_y)
            predict_test_y = model.predict(test_x)
            results_test = ErrorMetrics(predict_test_y * max_value, test_y * max_value)
            self.log(f"测试集上的表现 - MAE={results_test['MAE']:.4f}, RMSE={results_test['RMSE']:.4f}, NMAE={results_test['NMAE']:.4f}, NRMSE={results_test['NRMSE']:.4f}")
            self.log(f"Acc = [1%={results_test['Acc'][0]:.4f}, 5%={results_test['Acc'][1]:.4f}, 10%={results_test['Acc'][2]:.4f}]  ")
            results_dict[name] = results_test
        return results_dict

def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)
    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(args)
    model.set_runid(runId)
    # Prepare the data for machine learning
    train_x, train_y = datamodule.train_tensor[:, :-1], datamodule.train_tensor[:, -1]
    valid_x, valid_y = datamodule.valid_tensor[:, :-1], datamodule.valid_tensor[:, -1]
    test_x, test_y = datamodule.test_tensor[:, :-1], datamodule.test_tensor[:, -1]
    max_value = datamodule.max_value
    results = model.machine_learning_model_train_evaluation(train_x, train_y, valid_x, valid_y, test_x, test_y, max_value)
    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for model_name, model_results in results.items():
            for metric_name, metric_value in model_results.items():
                if metric_name == 'Acc':
                    continue
                    # for acc_name, acc_value in zip(['1', '5', '10'], metric_value):
                    #     metrics[f"{model_name}_{metric_name}_{acc_name}"].append(acc_value)
                else:
                    metrics[f"{model_name}_{metric_name}"].append(metric_value)
    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    if args.record:
        log.save_result(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20 + '\n')

    return metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='cpu')  #
    parser.add_argument('--model', type=str, default='CF')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.01)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=0)
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dimension', type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_settings(args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)



