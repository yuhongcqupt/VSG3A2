import csv
import math
import random
import numpy as np
import time
from sklearn.model_selection import KFold
import datetime

from monte_Carlo import MonteCarlo
from column_Distribution import ColumnDistribution
from entropy import Entropy
from code_convert import convert_float_to_754, convert_754_to_float

import random

import sys
import loadData


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w", encoding='utf-8')  #

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("..\Data\log.txt")  # 控制台输出信息保存到Data文件夹下
"""
流程；
    1 原始小样本即为原始种群，进行编码
"""


class model(object):
    """
    函数情况：
        1，选择函数，选择亲代样本
    """

    def __init__(self, data):
        """
        初始化原始数据
        :param data:原始数据集，保留三位小数的浮点数字符串 的ndarray二维数组形式
        """
        self.data = data
        self.parent1 = None  # 存放data中的一行，格式未变，还是三位小数的字符串
        self.parent2 = None
        self.parent1_chrom = None  # 统统使用ndarray形式
        self.parent2_chrom = None
        self.offspring1_chrom = None  # 统统使用ndarray形式
        self.offspring2_chrom = None
        self.a_virtual_sample = None
        self.offspring_float_str_after_mutation = {}  # 存放变异完之后的子代，是一个字典
        self.parent1_index = None
        self.parent2_index = None
        self.monte_carlo = []  # 存放各个属性列montecarlo模拟结果
        for attribute_chrom_index in range(len(data[0])):
            column_data = data.astype(float)[:, attribute_chrom_index]
            column_distribution = ColumnDistribution(column_data)
            column_distribution.fit_norm()  # 去拟合这个高斯分布
            monte_carlo_tmp = MonteCarlo(column_distribution)
            self.monte_carlo.append(monte_carlo_tmp)

    def select(self):
        """
        选择两个亲代样本
        选择方法：
            随机选择，因为初始数据集不能计算每条原始样本的信息增益
        :return:
        """
        while True:  # 不允许两条亲代样本都是
            self.parent1_index = np.random.randint(0, len(self.data))
            self.parent2_index = np.random.randint(0, len(self.data))
            if self.parent1_index != self.parent2_index:
                self.parent1 = self.data[self.parent1_index]
                self.parent2 = self.data[self.parent2_index]
                break

    def code(self):
        """
        编码方法：
            IEEE754编码方法
        :return:
        """
        parent_chrom = []
        for attribute in self.parent1:
            parent_chrom.append(convert_float_to_754(float(attribute)))
        self.parent1_chrom = np.array(parent_chrom)
        for attribute in self.parent2:
            parent_chrom.append(convert_float_to_754(float(attribute)))
        self.parent2_chrom = np.array(parent_chrom)

    def decode(self, row_binary_str):
        """
        row_str:一行，元素是字符串
        解码要用到的地方就是生成子代样本之后，需要解码，方便计算适应度值，但是计算的时候也是整个数据集一块输入进去的，所以还是先解码成字符串形式
        解码方法：
            IEEE754转浮点数
        :return:
        """
        row_float_str = []
        for attribute_binary_str in row_binary_str:
            attribute_float_str = str(convert_754_to_float(attribute_binary_str))
            row_float_str.append(attribute_float_str)
        return np.array(row_float_str)

    def crossover(self):
        """
        交叉算子：
            对于每个属性，先找到最小的字符串长度，因为属性值的编码结果万一不等长，应当以小的那个为准
            防止越界

        交叉完后，不转ndarray了，因为后面在mutation中需要求索引
        :return:
        """
        # print("进行交叉。")
        offspring1_chrom = []
        offspring2_chrom = []
        success = 1
        for attribute_index in range(len(self.parent1_chrom)):
            try_number = 0
            monte_carlo = self.monte_carlo[attribute_index]
            while try_number <= 1000:  # ------------------
                try_number += 1
                cross_point = np.random.randint(0, 32)  #
                offspring1_chrom_attribute = self.parent1_chrom[attribute_index][:cross_point + 1] + self.parent2_chrom[
                                                                                                         attribute_index][
                                                                                                     cross_point + 1:]
                offspring2_chrom_attribute = self.parent2_chrom[attribute_index][:cross_point + 1] + self.parent1_chrom[
                                                                                                         attribute_index][
                                                                                                     cross_point + 1:]
                offspring1_float_attribute = convert_754_to_float(offspring1_chrom_attribute)
                offspring2_float_attribute = convert_754_to_float(offspring2_chrom_attribute)
                if monte_carlo.is_retain(offspring1_float_attribute) and monte_carlo.is_retain(
                        offspring2_float_attribute):
                    offspring1_chrom.append(offspring1_chrom_attribute)
                    offspring2_chrom.append(offspring2_chrom_attribute)
                    break
                else:
                    continue
            if try_number > 1000:
                # print("本轮循环中，编号 " + str(attribute_index) + " 的属性交叉失败，本轮样本生成过程失效！")
                success = 0
                break
            else:
                self.offspring1_chrom = offspring1_chrom
                self.offspring2_chrom = offspring2_chrom
        return success

    def __mutation_one_offspring(self, offspring_chrom, num_of_offspring):
        """
        变异一条子代
        对于每一个属性列：
                取随机数 ，若小于0.05的变异概率，则变异：
                    原始样本数据集中的这一列的属性值提出来，去拟合高斯分布
                    对子代样本中的这个属性随机变异
                    变异完后要返回成浮点数的形式，
                    蒙特卡洛方法确定去留，若留，则continue
                    若弃，则重新选择变异点，重复上面步骤，直到可以留了。
                若大于0.05，则不变异：
                    continue
            变异完后，蒙特卡洛方法判断去留，若去，则重新变异
        :param offspring_chrom:输入是交叉后的子代，二进制格式的字符串的数组
        :return: 变异完成后解码，再返回一个解码后的浮点数字符串的数组
        """
        offspring_float_str_after_mutation = []

        success = 1  # 默认能够变异成功
        for attribute_chrom in offspring_chrom:
            rand_key = np.random.random(1)[0]  # 先判断是否需要变异
            mutation_rate = 0.01  # 因为总共就生成那么不到一百条，0.05的变异率对100来说太小了。0.001可
            try_num = 0
            if rand_key > mutation_rate:
                offspring_float_str_after_mutation.append(attribute_chrom)
                continue
            else:  # 若需要，则先学习属性分布，再进行变异，最后决定去留
                # print("本轮循环中的第" + str(num_of_offspring) + "条子代的编号" + str(
                #     offspring_chrom.index(attribute_chrom)) + "的属性需要变异！")
                attribute_index = offspring_chrom.index(attribute_chrom)
                monte_carlo = self.monte_carlo[attribute_index]
                while True:
                    if try_num <= 1000:
                        mutation_point = np.random.randint(0, 32)
                        attribute_chrom_list = [i for i in attribute_chrom]
                        if attribute_chrom_list[mutation_point] == '1':
                            attribute_chrom_list[mutation_point] = '0'
                        else:
                            attribute_chrom_list[mutation_point] = '1'
                        attribute_chrom_after_muntation = ''.join(attribute_chrom_list)
                        attribute_float_after_mutation = convert_754_to_float(attribute_chrom_after_muntation)
                        if monte_carlo.is_retain(attribute_float_after_mutation):  # 若符合保留条件，结束变异，否则重新变异
                            # print("变异成功！")
                            offspring_float_str_after_mutation.append(attribute_chrom_after_muntation)
                            break
                        else:
                            try_num += 1
                    else:
                        # print("本轮循环中的第" + str(num_of_offspring) + "条子代的编号" + str(
                        #     offspring_chrom.index(attribute_chrom)) + "的属性变异失败，本轮样本生成过程失效！")
                        success = 0  # 只要有一个属性列变异失败，那么总体的本轮生成过程就算失败
                        break
        if success:
            self.offspring_float_str_after_mutation[
                num_of_offspring] = offspring_float_str_after_mutation  # 不好返回，就直接存到类里面
        else:
            self.offspring_float_str_after_mutation = {}
        return success

    def mutation(self):
        """
        分别变异两条子代
            进行fitness计算，这两个子代中选择哪一个作为这一轮的虚拟样本。
            计算原始样本的信息熵
            分别计算添加了各个子代后的信息熵
            选择信息增益绝对值最小的那条子代作为本轮生成的虚拟样本
            原因:
                信息增益的方式，原理上是说得通的，但是实际上，即使在原始数据集上的测试，在增加了一条新样本后，整体的信息熵也不是都减少的。
                而是有增有减的，因此只能是选择信息增益的绝对值来判断。
        :return:
        """
        offspring1_chrom = self.offspring1_chrom
        offspring2_chrom = self.offspring2_chrom
        success = 0
        mutation_offspring1_success = self.__mutation_one_offspring(offspring1_chrom, 1)
        mutation_offspring2_success = self.__mutation_one_offspring(offspring2_chrom, 2)
        if mutation_offspring1_success and mutation_offspring2_success:  # 判断是否全都变异成功
            offspring1_float_after_mutation = self.decode(self.offspring_float_str_after_mutation[1])
            offspring2_float_after_mutation = self.decode(self.offspring_float_str_after_mutation[2])
            entropy = Entropy()  # 开始计算子代的适应度值
            data = self.data
            entropy_source = entropy.knn_entropy(data)
            data_add_offspring1 = np.append(data, [offspring1_float_after_mutation], axis=0)
            data_add_offspring2 = np.append(data, [offspring2_float_after_mutation], axis=0)
            entropy_add_offspring1 = entropy.knn_entropy(data_add_offspring1)
            entropy_add_offspring2 = entropy.knn_entropy(data_add_offspring2)
            entropy_gain_abs1 = abs(entropy_source - entropy_add_offspring1)
            entropy_gain_abs2 = abs(entropy_source - entropy_add_offspring2)
            if entropy_gain_abs1 > entropy_gain_abs2:  # 选择信息增益绝对值最小的子代作为本轮的虚拟样本
                self.a_virtual_sample = offspring1_float_after_mutation
            else:
                self.a_virtual_sample = offspring2_float_after_mutation
            success = 1
        return success

    def run(self, virtual_sample_size):
        """
        gavsg类的运行函数，参数为要运行的次数，也就是主程序中指定的生成虚拟样本的条数
        :param run_time:
        :return: 返回值是ndarray，每行元素是一条虚拟样本，行数就是run_time值
        """
        virtual_samples = []
        round_time = 0
        generate_time = 0
        while generate_time < virtual_sample_size:
            round_time += 1
            self.select()
            # print("第" + str(generate_time) + "轮循环--\n选择的亲代样本编号：" + str(self.parent1_index) + "," + str(
            #     self.parent2_index))
            self.code()
            success = self.crossover()
            if not success:
                continue
            success = self.mutation()
            if success:
                virtual_samples.append(self.a_virtual_sample)
                # print("第" + str(generate_time) + "遍生成完成!")
                # print(self.a_virtual_sample)
                generate_time += 1
            else:
                continue
        # print("本次程序执行共生成 " + str(generate_time) + " 条虚拟样本，共执行循环 " + str(round_time) + "次。")
        return np.array(virtual_samples)


import torch

if __name__ == '__main__':

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)  # 为CPU设置随机种子
    torch.cuda.manual_seed(1)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子

    starttime = datetime.datetime.now()

    for train_size in [10, 15, 20, 25, 30]:
        for rounder in range(1, 21):
            print("====train_size: " + str(train_size) + "=====round: " + str(
                rounder) + "=================================data generation=====")
            train_data_path = "..\\data\\data_division_" + str(train_size) + "+100\\ex1_ddfo_round_" + str(
                rounder) + "_data_10_train.csv"
            train_data = loadData.readData(train_data_path)

            tmp_model = model(train_data)
            virtual_samples = tmp_model.run(100)
            np.savetxt("..\\data\\data_division_" + str(train_size) + "+100\\ex1_ddfo_round_" + str(rounder) + "_data_10_train+100.csv",
                       np.vstack([virtual_samples.astype(float), train_data.astype(float)]), delimiter=',')

    endtime = datetime.datetime.now()
    print("程序运行时间：")
    print((endtime - starttime).seconds)
