# -*- coding: utf-8 -*
# @Time: 2021/5/16 14:11
from svmutil import *

y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]  # 输入的数据
options = '-t 0 -c 4 -b 1'  # 训练参数设置
model = svm_train(y, x, options)  # 进行训练

yt = [1]
xt = [{1: 1, 2: 1}]
p_label, p_acc, p_val = svm_predict(yt, xt, model)  # 使用得到的模型进行预测
print(p_label)
