import csv
import math
import random
from collections import namedtuple

import numpy as np
import open3d as o3d
from matplotlib import collections
import pandas as pd
import os
from sklearn.utils import shuffle
import numpy as np
import random

# data = pd.read_csv('data/my-exp2/myTrainData.csv', sep=',')
# data = shuffle(data)  # 打乱
# data.to_csv('data/my-exp2/myShuffleEvalData.csv', index=False, header=True)  # index索引不出现，header表头出现
#
data = pd.read_csv('data/my-exp7/avenue_raw.csv',nrows =10000)
data = shuffle(data)  # 打乱
data.to_csv('data/my-exp7/avenue.csv', index=False, header=True)



# def generate_point():
#     phi = random.uniform(0, 2 * math.pi)
#     theta = np.arccos(random.uniform(-1, 1))
#     return (theta, phi)
# if __name__ == '__main__':
#    print(generate_point())

import math

# R = 6378.1 #Radius of the Earth
# brng = 1.57 #Bearing is 90 degrees converted to radians.
# d = 15 #Distance in km
#
# #lat2  52.20444 - the lat result I'm hoping for
# #lon2  0.36056 - the long result I'm hoping for.
#
# lat1 = math.radians(52.20472) #Current lat point converted to radians
# lon1 = math.radians(0.14056) #Current long point converted to radians
#
# lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
#      math.cos(lat1)*math.sin(d/R)*math.cos(brng))
#
# lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
#              math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
#
# lat2 = math.degrees(lat2)
# lon2 = math.degrees(lon2)
#
# print(lat2)
# print(lon2)

# import pandas as pd
# # df = pd.read_csv("data/my-exp4/library_raw.csv")
# # print(min(df['lon']))
# # print(min(df['lat']))

# 计算以下数据的协方差矩阵
# import numpy as np
# import matplotlib.pyplot as plt  # Pyplot 是 Matplotlib 的子库，提供了和 MATLAB 类似的绘图 API。
#
# # 不用随机数了，这次用一个比较实际的例子，来测试PCA准不准
# # 假设这是一个扁扁的矩形,输入散点数据
# input = list()
# input.append([100, 100])
# input.append([200, 300])
#
# input.append([100, 400])
# input.append([400, 200])
# print(input)
#
# # 从已有的数组创建数组
# data = np.asarray(input)
# print(data)
#
# # 去中心化
# data_norm = data - data.mean(axis=0)  # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
#
# X = data_norm[:, 0]
# Y = data_norm[:, 1]
#
# C = np.cov(data_norm, rowvar=False)  ## 此时列为变量计算方式 即X为列，Y也为列
#
# # 计算特征值和特征向量
# # 这个返回的特征向量，要按列来看——因为，矩阵乘法来变换坐标系的时候，新基就是竖着的嘛
# vals, vecs = np.linalg.eig(C)
#
# # 重新排序，从大到小
# # 默认从小到大排列,这里取负了，就是从大到小排列，返回相应索引
# vecs = vecs[:, np.argsort(-vals)]  # 特征向量按列取，这里就呼应上了
# vals = vals[np.argsort(-vals)]
#
# print(vals)
# print(vecs)
#
# # 第一个特征值对应的特征向量
# print(vals[0], vecs[:, 0])
# # 第二个特征值对应的特征向量
# print(vals[1], vecs[:, 1])
#
# # 计算模长是否为1
# print(np.linalg.norm(vecs[:, 0]))
# print(np.linalg.norm(vecs[:, 1]))
#
# # 用画图的方式，画出结果
#
# size = 600  # 设置图大小
# plt.figure(1, (6, 6))
# plt.scatter(data[:, 0], data[:, 1], label='origin data')  # 使用 pyplot 中的 scatter() 方法来绘制散点图。
#
# # 逐个绘制方向向量
# i = 0
# ev = np.array([vecs[:, i] * -1, vecs[:, i]]) * size  # vecs竖着看的，是单位向量，取反是为了和正的构成两个点，绘制直线
# ev = (ev + data.mean(0))
# plt.plot(ev[:, 0], ev[:, 1], label='eigen vector ' + str(i + 1))  # ev是向量，竖着分离出向量的Xs和Ys，用来plot画图
#
# i = 1
# ev = np.array([vecs[:, i] * -1, vecs[:, i]]) * size
# ev = (ev + data.mean(0))
# plt.plot(ev[:, 0], ev[:, 1], label='eigen vector ' + str(i + 1))
#
# # 计算并绘制包围盒
#
# # 将原基下的坐标，变换到新基下，在新基下求包围盒
# Y = np.matmul(data_norm, vecs)  # 4*2 2*2 =4*2    #不出意外的话，这是新基下的坐标
# offset = 10
# xmin = min(Y[:, 0]) - offset  # 第一次独立写出python切片
# xmax = max(Y[:, 0]) + offset
# ymin = min(Y[:, 1]) - offset
# ymax = max(Y[:, 1]) + offset
#
# # 新基下的包围盒坐标
#
# temp = list()
#
# temp.append([xmin, ymin])
# temp.append([xmax, ymin])
# temp.append([xmax, ymax])
# temp.append([xmin, ymax])
#
# pointInNewCor = np.asarray(temp)
# # 将新基下计算出来的包围盒坐标，变换到原基下
# OBB = np.matmul(pointInNewCor, vecs.T) + data.mean(0)
#
# # 绘制包围盒
# plt.plot(OBB[0:2, 0], OBB[0:2, 1],
#          OBB[1:3, 0], OBB[1:3, 1],
#          OBB[2:4, 0], OBB[2:4, 1],
#          OBB[0:4:3, 0], OBB[0:4:3, 1],  # 最后一个切片，通过设置间隔，取开头和末尾
#          c='r'
#          )
#
# # 画一下x轴y轴
# plt.plot([-size, size], [0, 0], c='black')  # 画一条线，起点，终点，颜色
# plt.plot([0, 0], [-size, size], c='black')
# plt.xlim(-size, size)
# plt.ylim(-size, size)
# plt.legend()
# plt.show()
