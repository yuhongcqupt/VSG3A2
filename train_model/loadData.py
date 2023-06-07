# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 20:05
# @Author  : FanXuekang
# @File    : loadData.py
# @Software：PyCharm

import numpy as np


def readData(filePath):
    with open(filePath, encoding='utf-8') as f:  # 打开文件
        data = np.loadtxt(f, float, delimiter=',', skiprows=0)  # 读取方式float类型，逗号间隔，跳过首行，
    data = [list(row) for row in data]
    for row in range(len(data)):
        for col in range(len(data[row])):
            data[row][col] = (format(data[row][col], ".3f"))     # 小数三位，整数部分现在不定
    return np.array(data)