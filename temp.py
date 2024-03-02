# coding : utf-8
# Author : yuxiang Zeng
import pandas as pd
import numpy as np
import torch
data = pd.read_excel('./datasets/数据预处理.xlsx')
data.columns
data = data.drop(columns='case')
data = data.drop(columns='Unnamed: 5')
# data = data.to_numpy()
print(data)