# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from data import get_data, get_all_data
from main import get_args
from utils.metrics import ErrorMetrics


def train_and_evaluate(train_x, train_y, val_x, val_y, test_x, test_y, max_value):
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

    for name, model in models.items():
        print('-' * 80)
        print(f"正在训练模型: {name}")
        if name in param_grids:
            best_score = float('inf')
            best_params = None
            for params in ParameterGrid(param_grids[name]):
                model.set_params(**params)
                model.fit(train_x, train_y)
                predictions = model.predict(val_x)
                score = mean_squared_error(val_y, predictions)
                if score < best_score:
                    best_score = score
                    best_params = params
            print(f"{name} 最佳参数: {best_params}")
            model.set_params(**best_params)
            model.fit(np.vstack((train_x, val_x)), np.concatenate((train_y, val_y)))
        else:
            model.fit(np.vstack((train_x, val_x)), np.concatenate((train_y, val_y)))

        predict_test_y = model.predict(test_x)
        results_test = ErrorMetrics(predict_test_y * max_value, test_y * max_value)
        print(f"测试集上的表现 - MAE={results_test['MAE']:.4f}, RMSE={results_test['RMSE']:.4f}, NMAE={results_test['NMAE']:.4f}, NRMSE={results_test['NRMSE']:.4f}")
        print(f"Acc = [1%={results_test['Acc'][0]:.4f}, 5%={results_test['Acc'][1]:.4f}, 10%={results_test['Acc'][2]:.4f}]")


if __name__ == '__main__':
    args = get_args()
    X_train, y_train, X_val, y_val, X_test, y_test, max_value = get_all_data(args)
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, max_value)
