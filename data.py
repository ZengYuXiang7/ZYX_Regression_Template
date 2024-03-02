# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def get_data(args):
    df = pd.read_excel('./datasets/数据预处理.xlsx')
    df = df.drop(columns='Unnamed: 5')
    df = df.drop(columns='case')
    df = np.array(df)
    X, Y = df[:, :-2], df[:, -2]
    max_value = Y.max()
    Y_scaled = Y / max_value
    xscaler = MinMaxScaler()
    X_scaled = xscaler.fit_transform(X)
    return X_scaled, Y_scaled, max_value


def get_train_valid_test_dataset(args):
    X_scaled, Y_scaled, max_value = get_data(args)

    # 重新组合tensor为编码后的第一列、归一化的X和编码后的y
    tensor_normalized = np.hstack((X_scaled, Y_scaled.reshape(-1, 1))).astype(np.double)

    # 划分数据集
    trainsize = int(len(tensor_normalized) * 0.8)
    validsize = int(len(tensor_normalized) * 0.1)

    traintensor = tensor_normalized[:trainsize]
    validStart = trainsize
    validtensor = tensor_normalized[validStart:validStart + validsize]
    testStart = validStart + validsize
    testtensor = tensor_normalized[testStart:]
    return traintensor, validtensor, testtensor, max_value

def get_all_data(args):
    traintensor, validtensor, testtensor, max_value = get_train_valid_test_dataset(args)
    X_train = traintensor[:, :-1]
    y_train = traintensor[:, -1].reshape(-1, 1).ravel()
    X_val = validtensor[:, :-1]
    y_val = validtensor[:, -1].reshape(-1, 1).ravel()
    X_test = testtensor[:, :-1]
    y_test = testtensor[:, -1].reshape(-1, 1).ravel()
    return X_train, y_train, X_val, y_val, X_test, y_test, max_value