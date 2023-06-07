# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 10:56
# @Author  : FanXuekang
# @File    : monte_Carlo.py
# @Software：PyCharm
import math
import numpy as np
from numpy import random

class MonteCarlo(object):
    """
    使用蒙特卡洛采样方法中的接受拒绝采样，
    建议分布就使用均匀分布
    初始化参数就是已经拟合好的分布 ColumnDistribution

    说明：
        因为建议分布要求C*g(x) >= f(x),C可以取任意值，而且x的范围又没有定死
        所以可以就用比f(x)最大值略大的一个常数来替代
        这里就取f(x)最大值的向上取整
    """
    def __init__(self, column_distribution):
        """
        传入的是训练好的ColumnDistribution对象
        :param norm_distribution:
        """
        self.column_distribution = column_distribution
        self.norm_max_pdf = column_distribution.max_pdf_value
        # self.c_dot_gx = math.ceil(self.norm_max_pdf)
        self.c_dot_gx = self.norm_max_pdf*1.001    # 直接向上取整太大了

    def is_retain(self, x_new):
        """
        输入的x_new 是一个float
        使用接受-拒绝采样，判断新值x_new是否保留
        :param x_new:
        :return:
        """
        fx = self.column_distribution.norm_distribution.pdf(x_new)
        u = random.uniform(0, 1)
        # print("x_new"+str(x_new))
        # print(fx / self.c_dot_gx)
        # print(u)
        if (fx / self.c_dot_gx) >= u:
            return True
        else:
            return False