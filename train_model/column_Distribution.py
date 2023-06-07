# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 10:12
# @Author  : FanXuekang
# @File    : columnDistribution.py
# @Software：PyCharm
from scipy.stats import norm
from fitter import Fitter


class ColumnDistribution(object):
    """
    目的：
        使用小样本中的一列数字，去拟合出一个norm（高斯）分布，不用返回这个高斯分布的pdf，
        但应当能过够计算给定值的pdf值
    函数：
        1，拟合高斯分布的函数
        2，计算pdf值的函数
    """

    def __init__(self, column_data):
        """"参数column_data是ndarray格式数据，一行（表示的是原始数据的一个列）"""
        self.column_data = column_data
        self.mu = None
        self.delta = None
        self.max_pdf_value = None
        self.norm_distribution = None


    def fit_norm(self):
        """使用column_data去拟合高斯分布
            返回拟合出的高斯分布
            """
        f = Fitter(self.column_data, distributions='norm')
        f.fit()
        self.mu = list(f.get_best().values())[0]['loc']  ## 台式机用这个
        self.delta = list(f.get_best().values())[0]['scale']
        # self.mu = list(f.get_best().values())[0][0] #笔记本中用这个
        # self.delta = list(f.get_best().values())[0][1]
        norm_distribution = norm(self.mu, self.delta)
        self.norm_distribution = norm_distribution
        self.max_pdf_value = self.norm_distribution.pdf(self.mu)
        # return norm_distribution

    def predict(self, x):
        """计算属性值x对应的pdf值"""
        return self.norm_distribution.pdf(x)
