import csv
import numpy
from svmutil import *

# 读取西瓜数据集3.0csv文件
Dataset = []
datareader = csv.reader(
    open('WaterMelon_3.0.csv',
         mode='r',
         encoding='utf-8'))
for data in datareader:
    Dataset.append(data)
Dataset = numpy.array(Dataset)

# 提取样本含糖率属性
Y = Dataset[1:18, [9]].astype('int32')
# 提取样本密度属性
X = Dataset[1:18, [7, 8]].astype('float64')

y = X[:, 1]
x = []
# 生成libsvm标准格式训练数据
for i in range(len(y)):
    dic = {}
    dic[1] = X[i, 0]
    x.append(dic)

# 高斯核参数设置
guass_options = '-s 3 -t 2 -c 1 -b 1'
# 训练高斯核SVR模型
model = svm_train(y, x, guass_options)
svm_save_model('./ML6_8_guass', model)
