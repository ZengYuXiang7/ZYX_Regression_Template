# coding : utf-8
import statsmodels.api as sm
from run_deeplearning import get_args
from data import experiment, DataModule
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.utils import set_settings, set_seed
global log

if __name__ == '__main__':
    # Initialize
    args = get_args()
    set_seed(2024)
    set_settings(args)
    log = Logger(args)
    args.log = log
    exper = experiment(args)
    datamodule = DataModule(exper, args)

    # Prepare the data for machine learning
    train_x, train_y = datamodule.train_tensor[:, :-1], datamodule.train_tensor[:, -1]
    valid_x, valid_y = datamodule.valid_tensor[:, :-1], datamodule.valid_tensor[:, -1]
    test_x, test_y = datamodule.test_tensor[:, :-1], datamodule.test_tensor[:, -1]
    max_value = datamodule.max_value

    # 为X添加常数项以拟合截距
    X_train_sm = sm.add_constant(train_x)
    X_test_sm = sm.add_constant(test_x)

    # 使用statsmodels拟合多元线性回归模型
    model = sm.OLS(train_y, X_train_sm).fit()

    # 打印模型的摘要信息
    print(model.summary())

    # 使用sklearn的LinearRegression模型来打印和比较
    from sklearn.linear_model import LinearRegression
    model_sklearn = LinearRegression()
    model_sklearn.fit(train_x, train_y)

    # 模型性能评估
    y_pred = model_sklearn.predict(X_test_sm[:, 1:])  # 忽略statsmodels添加的常数项

    # 正确地进行反归一化处理
    y_test_inverse = (test_y * max_value).reshape(-1, 1).ravel()  # 反归一化
    y_pred_inverse = (y_pred * max_value).reshape(-1, 1).ravel()  # 反归一化

    # 直接使用反归一化后的数据计算误差指标
    results = ErrorMetrics(y_test_inverse, y_pred_inverse)

    # 输出模型性能指标
    log(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f}')
    log(f"Acc = [1%={results['Acc'][0]:.4f}, 5%={results['Acc'][1]:.4f}, 10%={results['Acc'][2]:.4f}]")

