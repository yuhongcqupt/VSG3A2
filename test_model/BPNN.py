import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import numpy as np  # numpy库
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, median_absolute_error, \
    r2_score
import random

import sys


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w", encoding='utf-8')  #

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("..\data\log_bpnn.txt")  # 控制台输出信息保存到Data文件夹下


class BPNN(object):
    def __init__(self, inputData, input_size, hidden_size, output_size, batch_size):  # 传入的都是float类型，ndarray类型，下面进行转换

        self.max = None
        self.min = None
        self.standardData = None

        self.inputData = inputData
        self.data = torch.tensor(inputData, dtype=float)
        self.x = inputData[:, :-1]
        self.y = inputData[:, -1]
        self.batch_size = batch_size  # 指定batchsize，对比算法中的神经网络就不用这个了，直接batchsize=1，这里是下游任务中的，所以指定
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.my_nn = None  # 保存训练好的网络模型

    def dataStandard(self):
        tmpMax = np.amax(self.inputData, axis=0)
        tmpMin = np.amin(self.inputData, axis=0)
        self.max = tmpMax
        self.min = tmpMin
        self.standardData = (self.inputData - self.min) / (self.max - self.min)
        return self.standardData

    def unStandard(self, standardY):
        sourceY = standardY * (self.max[-1] - self.min[-1]) + self.min[-1]
        return sourceY

    def train(self, iterationNum):
        self.dataStandard()
        # 搭建网络模型
        my_nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size, self.output_size),

            # torch.nn.Sigmoid(),
            # torch.nn.Linear(64, 128),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(128, 256),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(256, 512),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(512, self.output_size),

        )
        cost = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)  # 指定学习器，学习率
        # optimizer = torch.optim.SGD(my_nn.parameters(), lr=0.1)
        losses = [0]
        i = 0
        while i < iterationNum:
            i += 1
            batch_loss = []
            for start in range(0, len(self.data), self.batch_size):
                end = start + self.batch_size if start + self.batch_size < len(self.data) else len(self.data)  # 划分batch
                xx = torch.tensor(self.standardData[start:end, :-1], dtype=torch.float,
                                  requires_grad=True)  # 将数据转换成torch支持的类型
                yy = torch.tensor(self.standardData[start:end, -1], dtype=torch.float, requires_grad=True)
                prediction = my_nn(xx)
                loss = cost(prediction, yy)  # 计算损失
                optimizer.zero_grad()  # 对梯度清零
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss.append(loss.data.numpy())
            if i % 10 == 0:  # 每隔10次迭代，输出一次当前的损失值
                # print("第",str(i).ljust(10),"次迭代误差：", str(np.sqrt(np.mean(batch_loss))).rjust(10))
                if abs(losses[-1] - np.sqrt(np.mean(batch_loss))) < 1E-6:
                    break
                else:
                    losses.append(np.sqrt(np.mean(batch_loss)))
        self.my_nn = my_nn

    def predict(self, test_x):
        standardTestX = (test_x - self.min[:-1]) / (self.max[:-1] - self.min[:-1])
        standardTestX = torch.tensor(standardTestX, dtype=torch.float, requires_grad=True)
        pre = self.my_nn(standardTestX).detach().numpy()
        return self.unStandard(pre)


def mean_absolute_percentage_error(pred_y, test_y):
    pred_y = np.array(pred_y)
    test_y = np.array(test_y)
    tmp = abs((pred_y - test_y) / test_y)
    return sum(tmp) / len(tmp)


import datetime

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)  # 为CPU设置随机种子
    torch.cuda.manual_seed(1)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子

    for train_size in [10, 15, 20, 25, 30]:
        mae_list = []
        rmse_list = []
        mape_list = []
        for rounder in range(1, 21):
            print("====train_size: " + str(train_size) + "=====round: " + str(
                rounder) + "====================================bpnn training==")
            test_data_path = "..\\data\\data_division_" + str(train_size) + "+100\\ex1_ddfo_round_" + str(
                rounder) + "_data_10_test.csv"
            with open(test_data_path, encoding='utf-8') as f:  # 打开文件
                # 读取方式float类型，逗号间隔，跳过首行，
                data = np.loadtxt(f, float, delimiter=',', skiprows=0)
            test_data = np.array([list(row) for row in data])

            train_data_path = "..\\data\\data_division_" + str(train_size) + "+100\\ex1_ddfo_round_" + str(
                rounder) + "_data_10_train+100.csv"
            with open(train_data_path, encoding='utf-8') as f:  # 打开文件
                # 读取方式float类型，逗号间隔，跳过首行，
                data = np.loadtxt(f, float, delimiter=',', skiprows=0)
            train_data = np.array([list(row) for row in data])
            np.random.shuffle(train_data)
            bpnn_tmp = BPNN(train_data, len(train_data[0]) - 1, 32, 1, 1)
            bpnn_tmp.train(200)
            predict_y = bpnn_tmp.predict(test_data[:, :-1]).reshape(-1)
            test_y = test_data[:, -1]
            mae_tmp = mean_absolute_error(predict_y, test_y)
            rmse_tmp = mean_squared_error(predict_y, test_y) ** 0.5
            mape_tmp = mean_absolute_percentage_error(predict_y, test_y)

            mae_list.append(mae_tmp)
            mape_list.append(mape_tmp)
            rmse_list.append(rmse_tmp)
        np.savetxt(
            "..\\data\\data_division_" + str(train_size) + "+100\\bpnn_output_ex1_ddfo_mae_rmse_mape_20_division_"+str(train_size)+"+100_gavsg.csv",
            np.array([mae_list, rmse_list, mape_list]), delimiter=',')
        print("train_size: " + str(train_size) + "的数据——保存好了")
    endtime = datetime.datetime.now()
    print("程序运行时间：")
    print((endtime - starttime).seconds)
