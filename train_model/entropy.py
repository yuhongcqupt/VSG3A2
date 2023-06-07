# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 14:53
# @Author  : FanXuekang
# @File    : entropy_Gain.py
# @Software：PyCharm
import numpy as np
from scipy.special import gamma, digamma


class Entropy(object):
    """
    用于计算信息增益，便于计算fitness函数
    """

    # def __init__(self, data):
    #     """
    #     输入的data是float类型
    #     data_cache是在计算信息增益的过程中，一步步添加进去的，慢慢增大的数据集合
    #     :param data:
    #     """
    #     self.data = data

    def __euclidean_distance(self, sample1, sample2):
        """
        计算两条样本之间的欧氏距离，sampole1,sample2都是ndarray类型的一行数组
        :param sample1:
        :param sample2:
        :return:
        """
        return np.sqrt(sum((sample1 - sample2) ** 2))

    def __get_dis_of_ks_neighbor_for_sample_i(self, sample, samples, k):
        """
        类私有函数，用于计算样本sample在样本集samples中与它的第k个邻居的距离
        sample 是在samples中的
        传入的参数sample,与samples要求是float类型
        :param samples:
        :param k:
        :return:
        """
        every_sample_dis_to_sample_i = []
        for s in samples:
            every_sample_dis_to_sample_i.append(self.__euclidean_distance(s, sample))
        every_sample_dis_to_sample_i_sorted = sorted(every_sample_dis_to_sample_i)
        if len(samples) <= k:
            dis_of_k_nei = every_sample_dis_to_sample_i_sorted[-1]  # 返回第k个距离值
        else:
            dis_of_k_nei = every_sample_dis_to_sample_i_sorted[k]  # 返回第k个距离值，自己与自己距离永远是最近的，但不能算上自己
        return dis_of_k_nei

    def __sum_of_log(self, samples, k):
        """
        k近邻无参数熵估计法公式的 求和部分的计算
        samples是float类型
        :param samples:
        :param k:
        :return:
        """
        sum = 0.0
        for sample in samples:
            dis_of_k_nei = self.__get_dis_of_ks_neighbor_for_sample_i(sample, samples, k)
            sum += np.log(dis_of_k_nei)
        return sum

    def knn_entropy(self, data):
        """
        计算数据集data的信息熵，在主程序中就先计算原始数据集的信息熵，再计算加入新样本之后的信息熵
        data是float类型的的ndarray二维数组
        :param data:
        :return:
        """
        dim = len(data[0])
        k = 3  # k取推荐值3
        n = len(data)

        cd = ((np.pi) ** (dim / 2)) / (gamma(1 + (dim / 2)))
        entropy = digamma(n) - digamma(k) + np.log(cd) + (dim / n) * (self.__sum_of_log(data.astype(float), k))  # 熵小
        return entropy

    # def caculate_entropy(self, row):
    #     """
    #     计算新加入这一行row数据，对样本集合
    #     本来是计算row这条样本带来的信息增益，主程序中对两条样本带来的信息增益比较，
    #     但是因为原始种群是保持不变的，所以原始种群的信息熵是保持不变的，因此这里只需要
    #     返回原始样本集在加入row之后的信息熵，将这个信息熵值返回，在出程序中直接判断即可。
    #     注意：
    #         因为是连续值，这里信息熵的计算公式采用 --无参数熵估计法中的k-近邻估计法
    #     :param row:
    #     :return:-加入row这条样本之后的信息熵
    #     """
    #     self.
    #     return 0

#
# import loadData
#
# if __name__ == '__main__':
#     data = loadData.readMLCC()
#     for rownum in range(30,43):
#
#         d = data.astype(float)[:rownum]
#         entropy = Entropy()
#         # entropy = Entropy(data)
#         en = entropy.knn_entropy(d)
#         print(en)


