# coding : utf-8
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from data import get_data
from main import get_args
from utils.metrics import ErrorMetrics


if __name__ == '__main__':
    args = get_args()
    X_scaled, Y_scaled, max_value = get_data(args)

    # 切分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)

    # 为X添加常数项以拟合截距
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    # 使用statsmodels拟合多元线性回归模型
    model = sm.OLS(y_train, X_train_sm).fit()

    # 打印模型的摘要信息
    print(model.summary())

    # 使用sklearn的LinearRegression模型来打印和比较
    from sklearn.linear_model import LinearRegression
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)

    # 模型性能评估
    y_pred = model_sklearn.predict(X_test_sm[:, 1:])  # 忽略statsmodels添加的常数项

    # 正确地进行反归一化处理
    y_test_inverse = (y_test * max_value).reshape(-1, 1).ravel()  # 反归一化
    y_pred_inverse = (y_pred * max_value).reshape(-1, 1).ravel()  # 反归一化

    # 直接使用反归一化后的数据计算误差指标
    results = ErrorMetrics(y_test_inverse, y_pred_inverse)

    # 输出模型性能指标
    print(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f}')
    print(f"Acc = [1%={results['Acc'][0]:.4f}, 5%={results['Acc'][1]:.4f}, 10%={results['Acc'][2]:.4f}]")

