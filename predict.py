# coding : utf-8
# Author : yuxiang Zeng


import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from data import experiment, DataModule
from get_params import get_ml_args
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.utils import set_settings, set_seed
global log


def get_trained_model(datamodule, args):
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import ParameterGrid
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    train_x, train_y = datamodule.train_tensor[:, :-1], datamodule.train_tensor[:, -1]
    valid_x, valid_y = datamodule.valid_tensor[:, :-1], datamodule.valid_tensor[:, -1]
    test_x, test_y = datamodule.test_tensor[:, :-1], datamodule.test_tensor[:, -1]
    max_value = datamodule.max_value
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'SVR': SVR(max_iter=1000000),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
    }
    param_grids = {
        'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
        'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 9, 11, 15]},
        'SVR': {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear']},
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

    global_best_model_name = None
    global_best_model = None
    global_best_score = float('inf')
    global_best_params = None
    for name, model in models.items():
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
            model.set_params(**best_params)
        model.fit(train_x, train_y)
        predict_test_y = model.predict(test_x)
        results_test = ErrorMetrics(predict_test_y * max_value, test_y * max_value)
        if args.experiment:
            log(f"测试集上的表现 - MAE={results_test['MAE']:.4f}, RMSE={results_test['RMSE']:.4f}, NMAE={results_test['NMAE']:.4f}, NRMSE={results_test['NRMSE']:.4f} ")
            log(f"Acc = [1%={results_test['Acc'][0]:.4f}, 5%={results_test['Acc'][1]:.4f}, 10%={results_test['Acc'][2]:.4f}]  ")
        else:
            if results_test['NMAE'] < global_best_score:
                global_best_model_name = name
                global_best_model = model
                global_best_score = results_test['NMAE']
                if name in param_grids:
                    global_best_params = best_params
        results_dict[name] = results_test
    if global_best_model is not None and not args.experiment:
        log(f"最佳模型: {global_best_model_name} 最佳参数: {global_best_params} 最佳NMAE: {global_best_score}")
        if global_best_model_name in param_grids:
            global_best_model.set_params(**global_best_params)
        global_best_model.fit(np.vstack((train_x, valid_x)), np.hstack((train_y, valid_y)))
        predict_test_y = global_best_model.predict(test_x)
        results_test = ErrorMetrics(predict_test_y * max_value, test_y * max_value)
        log(f"测试集上的表现 - MAE={results_test['MAE']:.4f}, RMSE={results_test['RMSE']:.4f}, NMAE={results_test['NMAE']:.4f}, NRMSE={results_test['NRMSE']:.4f}")
        log(f"Acc = [1%={results_test['Acc'][0]:.4f}, 5%={results_test['Acc'][1]:.4f}, 10%={results_test['Acc'][2]:.4f}]")
    return results_dict, global_best_model



if __name__ == '__main__':
    # Initialize
    args = get_ml_args()
    log = Logger(args)
    args.log = log

    # Train the model
    set_seed(2)
    args.index = 0
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    _, model = get_trained_model(datamodule, args)

    ratio1_1, ratio1_2 = 43, 23
    X = np.array([ratio1_1, ratio1_2]).reshape(-1, 2)
    Y = model.predict(X)

