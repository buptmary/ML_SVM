# -*- coding: utf-8 -*
# @Time: 2021/5/16 20:31
import numpy
import csv
from svmutil import *

# 读取西瓜数据集3.0csv文件
Dataset = []
datareader = csv.reader(
    open(r'WaterMelon_3.0.csv',
         mode='r',
         encoding='utf-8'))
for data in datareader:
    Dataset.append(data)
Dataset = numpy.array(Dataset)

# 提取样本label并转换为1、-1
Y = Dataset[1:18, [9]].astype('int32')
Y = 2 * Y - 1
# 提取样本密度、含糖率
X = Dataset[1:18, [7, 8]].astype('float64')

y = Y.flatten()  # flatten()用于矩阵降维
x = []
# 生成libsvm标准格式训练数据
for i in range(len(y)):
    dic = {}
    dic[1] = X[i, 0]
    dic[2] = X[i, 1]
    x.append(dic)

# 线性核参数设置
linear_options = '-t 0 -c 1 -b 1'
# 高斯核参数设置
gaosi_options = '-t 2 -c 4 -b 1'

# 训练线性核svm模型
linear_model = svm_train(y, x, linear_options)
svm_save_model('./ML6_2_linear', linear_model)
# 训练高斯核svm模型
gauss_model = svm_train(y, x, gaosi_options)
svm_save_model('./ML6_2_gauss', gauss_model)
