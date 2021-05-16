# -*- coding: utf-8 -*
# @Time: 2021/5/13 13:20

import pandas as pd
from svmutil import *


# 将西瓜数据集3.0a转化为LIBSvm包规定的数据格式
def process_data(filename):
    df = pd.read_csv(filename, sep=' ')
    data = df.values
    x1 = data[:, 1]
    y = data[:, 2]
    with open('watermelon_3a_svr.txt', 'w') as f:
        for m1, m2 in zip(x1, y):
            f.write("{} 1:{}\n".format(round(m2, 3), round(m1, 3)))
    f.close()


def main():
    process_data('watermelon_3a.txt')
    y, x = svm_read_problem('watermelon_3a_svm.txt')
    model3 = svm_train(y, x, '-t 3 -c 100 -g 10')  # 线性核
    svm_save_model('SVR.model', model3)
    svm_predict(y, x, model3)


if __name__ == '__main__':
    main()
