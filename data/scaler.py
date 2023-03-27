from typing import Any, List

import numpy as np

# 归一化
class StandardScaler:
    """
    StandardScaler 规范化数据集。

     当适合数据集时，StandardScaler 学习第 0 轴上的平均值和标准偏差。
     转换数据集时，StandardScaler 减去均值并除以标准差。
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        初始化 StandardScaler，可选择预先计算均值和标准差。

         :param 表示：一个可选的一维 numpy 预计算平均值数组。
         :param stds: 一个可选的一维 numpy 预计算标准差数组。
         :param replace_nan_token: 用来代替 nans 的令牌。
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[float]]) -> 'StandardScaler':
        """
        学习第 0 轴上的均值和标准差。

         :param X: 浮点数列表。
         :return: 适合的 StandardScaler。
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[float]]):
        """
        通过减去均值并除以标准差来转换数据。

         :param X: 浮点数列表。
         :return: 转换后的数据。
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[float]]):
        """
        通过乘以标准差并添加均值来执行逆变换。

         :param X: 浮点数列表。
         :return: 逆变换的数据。
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
