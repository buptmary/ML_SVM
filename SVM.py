# -*- coding: utf-8 -*
# @Time: 2021/5/13 13:20

import pandas as pd
from svmutil import *


# 将西瓜数据集3.0a转化为LIBSvm包规定的数据格式
def process_data(filename):
    df = pd.read_csv(filename, sep=' ')
    data = df.values
    x1 = data[:, 1]
    x2 = data[:, 2]
    y = data[:, 3]
    with open('watermelon_3a_svm.txt', 'w') as f:
        for m1, m2, n in zip(x1, x2, y):
            f.write("{} 1:{} 2:{}\n".format(int(n), round(m1, 3), round(m2, 3)))
    f.close()


def main():
    process_data('watermelon_3a.txt')
    y, x = svm_read_problem('watermelon_3a_svm.txt')
    model1 = svm_train(y, x, '-t 0 -c 1000')  # 线性核
    model2 = svm_train(y, x, '-t 2 -c 1000 -g 0.8')  # 高斯核
    svm_save_model('Linear.model', model1)
    svm_save_model('Gauss.model', model2)
    svm_predict(y, x, model2)


if __name__ == '__main__':
    main()
