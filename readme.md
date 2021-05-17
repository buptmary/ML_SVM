# **目录** {#目录 .TOC-Heading}

[**1、数据集介绍** 2](#_Toc67844216)

[**2、SVM支持向量机模型** 2](#_Toc67844217)

> [**2.1、支持向量机模型介绍** 2](#_Toc67844218)
>
> [**2.2、支持向量机SVM算法** 5](#_Hlk69070090)

[**3、习题解答** 6](#_Toc72080169)

> [**3.1、习题6.2** 6](#_Toc72080170)
>
> [**3.2、习题6.8** 9](#_Toc72080171)

[**4、附录** 11](#_Toc72080172)

# 第六章 支持向量机

[]{#_Toc67844216 .anchor}**1、数据集介绍**

本次实验使用到一个数据集，为西瓜数据集3.0$\alpha$。西瓜数据集3.0$\alpha$包含17条信息，每条信息对应西瓜的2种属性，给出了该西瓜是否为好瓜，"是"表示该西瓜是好瓜，"否"表该西瓜不是好瓜。西瓜数据集3.0$\alpha$的具体内容如下图所示。

表1 西瓜数据集3.0$\alpha$

![](media/image1.png){width="3.713542213473316in" height="4.2962959317585305in"}

[]{#_Toc67844217 .anchor}**2、SVM支持向量机模型**

[]{#_Toc67844218 .anchor}**2.1、支持向量机模型介绍**

支持向量机（support vector machines, SVM）是一类按监督学习方式对数据进行二分类的广义线性分类器，其决策边界是对学习样本求解的最大边距超平面，间隔最大使它有别于感知机；SVM还包括核技巧，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。SVM的的学习算法就是求解凸二次规划的最优化算法。

![](media/image2.jpeg){width="3.34375in" height="2.944803149606299in"}

图2.1.1 超平面模型

SVM学习的基本想法是求解能够正确划分训练数据集并且几何间隔最大的分离超平面。如下图所示， $\mathbf{w} \cdot x + b = 0$即为分离超平面，对于线性可分的数据集来说，这样的超平面有无穷多个（即感知机），但是几何间隔最大的分离超平面却是唯一的。

**·间隔最大化**

假设给定一个特征空间上的训练数据集$T = \left\{ \left( \mathbf{x}_{1},y_{1} \right),\left( \mathbf{x}_{2},y_{2} \right),\ldots,\left( \mathbf{x}_{N},y_{N} \right) \right\}$，其中，$\mathbf{x}_{i} \in \mathbb{R}^{n},y_{i} \in \{ + 1, - 1\},i = 1,2,\ldots N$，$\mathbf{x}_{i}$为第$i$个特征向量，$y_{i}$为类标记，当它等于+1时为正例；为-1时为负例。再假设训练数据集是线性可分的。

对于给定的数据集$T$和超平面$\mathbf{w} \cdot x + b = 0$，定义超平面关于样本点$\left( \mathbf{x}_{i},y_{i} \right)$ 的几何间隔为:

$$\gamma_{i} = y_{i}\left( \frac{\mathbf{w}}{\parallel \mathbf{w} \parallel} \cdot \mathbf{x}_{i} + \frac{b}{\parallel \mathbf{w} \parallel} \right)$$

超平面关于所有样本点的几何间隔的最小值为:

$$\gamma = \min_{i = 1,2\ldots,N}\mspace{2mu}\gamma_{i}$$

根据以上定义，SVM模型的求解最大分割超平面问题可以表示为以下约束最优化问题:

$$\min_{\mathbf{w},b}\mspace{2mu}\frac{1}{2} \parallel \mathbf{w} \parallel^{2}$$

$$\text{s.t.}\text{\ }y_{i}\left( \mathbf{w} \cdot \mathbf{x}_{i} + b \right) \geq 1,i = 1,2,\ldots,N$$

**·对偶问题**

这是一个含有不等式约束的凸二次规划问题，可以对其使用拉格朗日乘子法得到其对偶问题（dual problem）。

首先，我们将有约束的原始目标函数转换为无约束的新构造的拉格朗日目标函数：

$$L(\mathbf{w},b,\mathbf{\alpha}) = \frac{1}{2} \parallel \mathbf{w} \parallel^{2} - \sum_{i = 1}^{N}\mspace{2mu}\alpha_{i}\left( y_{i}\left( \mathbf{w} \cdot \mathbf{x}_{\mathbf{i}} + b \right) - 1 \right)$$

根据拉格朗日函数对偶性，要满足对偶性，需要满足：①优化问题是凸优化问题；②满足KKT条件。为了得到求解对偶问题的具体形式，令$L(\mathbf{w},b,\mathbf{\alpha})$对 $\mathbf{w}$和$b$的偏导为0，再带入拉格朗日目标函数，消去$\mathbf{w}$和$b$，得：

$$\begin{matrix}
L(\mathbf{w},b,\mathbf{\alpha}) = \frac{1}{2}\sum_{i = 1}^{N}\mspace{2mu}\sum_{j = 1}^{N}\mspace{2mu}\alpha_{i}\alpha_{j}y_{i}y_{j}\left( \mathbf{x}_{\mathbf{i}} \cdot \mathbf{x}_{\mathbf{j}} \right) - \sum_{i = 1}^{N}\mspace{2mu}\alpha_{i}y_{i}\left( \left( \sum_{j = 1}^{N}\mspace{2mu}\alpha_{j}y_{j}\mathbf{x}_{\mathbf{j}} \right) \cdot \mathbf{x}_{\mathbf{i}} + b \right) + \sum_{i = 1}^{N}\mspace{2mu}\alpha_{i} \\
 = - \frac{1}{2}\sum_{i = 1}^{N}\mspace{2mu}\sum_{j = 1}^{N}\mspace{2mu}\alpha_{i}\alpha_{j}y_{i}y_{j}\left( \mathbf{x}_{i} \cdot \mathbf{x}_{\mathbf{j}} \right) + \sum_{i = 1}^{N}\mspace{2mu}\alpha_{i} \\
\end{matrix}$$

原问题转换为以下优化问题：

$$\begin{matrix}
\&\min_{\mathbf{\alpha}}\mspace{2mu}\frac{1}{2}\sum_{i = 1}^{N}\mspace{2mu}\sum_{j = 1}^{N}\mspace{2mu}\alpha_{i}\alpha_{j}y_{i}y_{j}\left( \mathbf{x}_{i} \cdot \mathbf{x}_{j} \right) - \sum_{i = 1}^{N}\mspace{2mu}\alpha_{i} \\
\&\text{\ s.t.\ }\sum_{i = 1}^{N}\mspace{2mu}\alpha_{i}y_{i} = 0 \\
\&\alpha_{i} \geq 0,i = 1,2,\ldots,N \\
\end{matrix}$$

对于任意训练样本$\left( \mathbf{x}_{i},y_{i} \right)$，总有$\alpha_{i} = 0$或者$y_{i}(\mathbf{w} \cdot x + b) = 0$。若$\alpha_{i} = 0$，则该样本不会在最后求解模型参数的式子中出现。若$\alpha_{i} > 0$，则必有$y_{i}(\mathbf{w} \cdot x + b) = 1$，所对应的样本点位于最大间隔边界上，是一个支持向量。这显示出支持向量机的一个重要性质：训练完成后，大部分的训练样本都不需要保留，最终模型仅与支持向量有关。

**·非线性可分与核函数**

以上讨论都是在样本完全线性可分或者大部分样本点线性可分，但我们可能会遇到线性不可分的情况。在这种情况下，将二维线性不可分样本映射到高维空间中，让样本点在高维空间线性可分。

![C:\\Users\\wbq\\Desktop\\QQ截图20140829153335.png](media/image3.png){width="5.768055555555556in" height="2.3541666666666665in"}

图2.1.2 线性不可分转化为线性可分

对于在有限维度向量空间中线性不可分的样本，我们将其映射到更高维度的向量空间里，再通过间隔最大化的方式，学习得到支持向量机。

我们用$\text{\ x\ }$表示原来的样本点，用$\phi(x)$表示$\text{\ x\ }$映射到特征新的特征空间后到新向量。那么分割超平面可以表示为：$f\left( x \right) = \mathbf{w} \cdot \phi(x) + b$。

对于非线性 SVM 的对偶问题就变成了：

$$\begin{matrix}
\&\min_{\lambda}\mspace{2mu}\left\lbrack \frac{1}{2}\sum_{i = 1}^{n}\mspace{2mu}\sum_{j = 1}^{n}\mspace{2mu}\lambda_{i}\lambda_{j}y_{i}y_{j}\left( \phi\left( x_{i} \right) \cdot \phi\left( x_{j} \right) \right) - \sum_{j = 1}^{n}\mspace{2mu}\lambda_{i} \right\rbrack \\
\&\text{\ s.t.\ }\sum_{i = 1}^{n}\mspace{2mu}\lambda_{i}y_{i} = 0,\lambda_{i} \geq 0,C - \lambda_{i} - \mu_{i} = 0 \\
\end{matrix}$$

由于特征空间的维数可能很高，甚至是无穷维，因此直接计算$\phi\left( x_{i} \right) \cdot \phi\left( x_{j} \right)$通常是困难的，于是引入了核函数$\kappa\left( x_{i},x_{j} \right) = < \phi\left( x_{i} \right),\phi\left( x_{j} \right) > = \phi\left( x_{i} \right)^{T}\phi\left( x_{j} \right)$。

[]{#_Hlk69070090 .anchor}**2.2、支持向量机SVM算法**

支持向量机学习算法如下：

**输入：**训练数据集$T = \left\{ \left( \mathbf{x}_{1},y_{1} \right),\left( \mathbf{x}_{2},y_{2} \right),\ldots,\left( \mathbf{x}_{N},y_{N} \right) \right\}$，其中，$\mathbf{x}_{i} \in \mathbb{R}^{n},y_{i} \in \text{\ \ \ \ \ \ \ }\{ + 1, - 1\},i = 1,2,\ldots N$;

**输出：**分离超平面和分类决策函数

(1)选取适当的核函数$K\left( x_{i},x_{j} \right)$和惩罚参数$C > 0$，构造并解决凸二次规划问题：

$$\begin{matrix}
\&\min_{\mathbf{\alpha}}\mspace{2mu}\frac{1}{2}\sum_{i = 1}^{N}\mspace{2mu}\sum_{j = 1}^{N}\mspace{2mu}\alpha_{i}\alpha_{j}y_{i}y_{j}K\left( \mathbf{x}_{\mathbf{i}},\mathbf{x}_{\mathbf{j}} \right) - \sum_{i = 1}^{N}\mspace{2mu}\alpha_{i} \\
\&\text{\ s.t.\ }\sum_{i}^{N}\mspace{2mu}\alpha_{i}y_{i} = 0 \\
\end{matrix}$$

得到最优解$\mathbf{\alpha}^{*} = \left( \alpha_{1}^{*},\alpha_{2}^{*},\ldots,\alpha_{N}^{*} \right)^{T}$

(2)计算$\mathbf{\alpha}^{*}$的一个分量$\alpha_{j}^{*}$满足条件$0 < \alpha_{j}^{*} < C$，计算

$\mathbf{b}^{*} = \mathbf{\alpha}^{*} - \sum_{i = 1}^{N}{\mathbf{\alpha}^{*}y_{i}K(x_{i}{,x}_{i})}$。

(3)分类决策函数：

$$f(x) = \text{sign}\left( \sum_{i = 1}^{N}\mspace{2mu}\alpha_{i}^{*}y_{i}K\left( x,x_{i} \right) + b^{*} \right)$$

[]{#_Toc72080169 .anchor}**3、习题解答**

[]{#_Toc72080170 .anchor}**3.1、习题6.2**

【习题5.5】试用LIBSVM，在西瓜数据集3.0$\alpha$上分别用线性核和高斯核训练一个SVM，并比较其支持向量的差别。

代码解析：

**·LIBSVM包的安装**

打开https://www.lfd.uci.edu/\~gohlke/pythonlibs/\#libsvm，下载python3.8对应的文件，在whl文件目录下执行pip install安装。

**·数据处理**

LIBSVM的数据格式为：\<label\> \<index1\>:\<value1\> \<index2\>:\<value2\> \...

第一列为标签；第二列为索引（1）和第一个特征值；第三列为索引（2）和第二个特征值；

代码如下：

1.  *\# 将西瓜数据集3.0a转化为LIBSvm包规定的数据格式*

2.  def process_data(filename):

3.      df = pd.read_csv(filename, sep=\' \')

4.      data = df.values

5.      x1 = data\[:, 1\]

6.      x2 = data\[:, 2\]

7.      y = data\[:, 3\]

8.      with open(\'watermelon_3a_svm.txt\', \'w\') as f:

9.          for m1, m2, n in zip(x1, x2, y):

10.             f.write(\"{} 1:{} 2:{}\\n\".format(int(n), round(m1, 3), round(m2, 3)))

11.     f.close()

生成的数据如下：

1.  1 1:0.697 2:0.46

2.  1 1:0.774 2:0.376

3.  1 1:0.634 2:0.264

4.  1 1:0.608 2:0.318

5.  1 1:0.556 2:0.215

6.  1 1:0.403 2:0.237

7.  1 1:0.481 2:0.149

8.  1 1:0.437 2:0.211

9.  0 1:0.666 2:0.091

10. 0 1:0.243 2:0.267

11. 0 1:0.245 2:0.057

12. 0 1:0.343 2:0.099

13. 0 1:0.639 2:0.161

14. 0 1:0.657 2:0.198

15. 0 1:0.36 2:0.37

16. 0 1:0.593 2:0.042

17. 0 1:0.719 2:0.103

**·主函数调用**

本次使用LIBSVM进行训练，所以只需要调用相应函数即可实现，svm_read_problem函数读取已经修改好格式的数据集，svm_train函数负责svm训练，可以对svm模型参数进行调整，训练好的模型可以通过svm_save_model进行文件存储，svm_predict函数负责通过已知模型预测输出。

1.  def main():

2.      process_data(\'watermelon_3a.txt\')

3.      y, x = svm_read_problem(\'watermelon_3a_svm.txt\')

4.      model1 = svm_train(y, x, \'-t 0 -c 1000\')  *\# 线性核*

5.      model2 = svm_train(y, x, \'-t 2 -c 900 -g 0.8\')  *\# 高斯核*

6.      svm_save_model(\'Linear.model\', model1)

7.      svm_save_model(\'Gauss.model\', model2)

8.      svm_predict(y, x, model2)

完整代码参见附录1，SVM.py

线性核和高斯核SVM输出如下图所示：

①线性核SVM，C = 1000， 准确率82.4%。

![](media/image4.png){width="4.882315179352581in" height="1.4947911198600174in"}

参数解读：

nu: 错误率nu参数

obj: SVM 文件转换为的二次规划求解得到的最小值

rho:为判决函数的常数项 b

nSV:为支持向量个数

nBSV: 边界上的支持向量个数

Total nSV:为支持向量总个数

另外，为了更加直观的看到SVM的分类情况，绘制了数据集的散点分布图，标出了支持向量以及决策边界：

![](media/image5.png){width="4.255905511811024in" height="3.1692913385826773in"}

②高斯性核SVM，C = 1000，gamma = 0.8 准确率100%。

![](media/image6.png){width="5.768055555555556in" height="1.6715277777777777in"}

参数解读：

nu: 错误率nu参数

obj: SVM 文件转换为的二次规划求解得到的最小值

rho:为判决函数的常数项 b

nSV:为支持向量个数

nBSV: 边界上的支持向量个数

Total nSV:为支持向量总个数

另外，为了更加直观的看到SVM的分类情况，绘制了数据集的散点分布图，标出了支持向量以及决策边界：

![](media/image7.png){width="4.291338582677166in" height="3.177165354330709in"}

线性核与高斯核的支持向量对比

![](media/image5.png){width="2.6141732283464565in" height="1.9488188976377954in"}![](media/image7.png){width="2.6338582677165356in" height="1.9488188976377954in"}

对比线性核与高斯核的支持向量，在二者C=1000的情况下，线性核的支持向量为12个，高斯核支持向量为8个，线性核比高斯核的支持向量要多，但是，从分类结果来看，高斯核的准确率更高。

[]{#_Toc72080171 .anchor}**3.2、习题6.8**

以西瓜数据集3.0$\alpha$的"密度"为输入，"含糖率"为输出，试用LIBSVM训练一个SVR。

代码解析：

**·LIBSVM包的安装**

打开https://www.lfd.uci.edu/\~gohlke/pythonlibs/\#libsvm，下载python3.8对应的文件，在whl文件目录下执行pip install安装。

**·数据处理**

LIBSVM的数据格式为：\<label\> \<index1\>:\<value1\> \<index2\>:\<value2\> \...

第一列为标签；第二列为索引（1）和第一个特征值；第三列为索引（2）和第二个特征值；

代码如下：

1.  *\# 将西瓜数据集3.0a转化为LIBSvm包规定的数据格式*

2.  def process_data(filename):

3.      df = pd.read_csv(filename, sep=\' \')

4.      data = df.values

5.      x1 = data\[:, 1\]

6.      y = data\[:, 2\]

7.      with open(\'watermelon_3a_svr.txt\', \'w\') as f:

8.          for m1, m2 in zip(x1, y):

9.              f.write(\"{} 1:{}\\n\".format(round(m2, 3), round(m1, 3)))

10.     f.close()

生成的数据如下

1.  0.46 1:0.697

2.  0.376 1:0.774

3.  0.264 1:0.634

4.  0.318 1:0.608

5.  0.215 1:0.556

6.  0.237 1:0.403

7.  0.149 1:0.481

8.  0.211 1:0.437

9.  0.091 1:0.666

10. 0.267 1:0.243

11. 0.057 1:0.245

12. 0.099 1:0.343

13. 0.161 1:0.639

14. 0.198 1:0.657

15. 0.37 1:0.36

16. 0.042 1:0.593

17. 0.103 1:0.719

**·主函数调用**

本次使用LIBSVM进行训练，所以只需要调用相应函数即可实现，svm_read_problem函数读取已经修改好格式的数据集，svm_train函数负责svm训练，可以对svm模型参数进行调整 -t 3即选择SVR模型，训练好的模型可以通过svm_save_model进行文件存储，svm_predict函数负责通过已知模型预测输出。

1.  def main():

2.      process_data(\'watermelon_3a.txt\')

3.      y, x = svm_read_problem(\'watermelon_3a_svm.txt\')

4.      model3 = svm_train(y, x, \'-t 3 -c 10 -g 8\')  *\# 线性核*

5.      svm_save_model(\'SVR.model\', model3)

6.      svm_predict(y, x, model3)

运行程序输出结果如下：

![](media/image8.png){width="5.768055555555556in" height="1.5298611111111111in"}

参数解读：

nu: 错误率nu参数

obj: SVM 文件转换为的二次规划求解得到的最小值

rho:为判决函数的常数项 b

nSV:为支持向量个数

nBSV: 边界上的支持向量个数

Total nSV:为支持向量总个数

从输出结果可以看到，最终有8个支持向量，模型准确率53%

[]{#_Toc72080172 .anchor}**4、附录**

1、SVM.py

+-------------------------------------------------------------------------+
| \# -\*- coding: utf-8 -\*                                               |
|                                                                         |
| \# \@Time: 2021/5/13 13:20                                              |
|                                                                         |
| import pandas as pd                                                     |
|                                                                         |
| from svmutil import \*                                                  |
|                                                                         |
| \# 将西瓜数据集3.0a转化为LIBSvm包规定的数据格式                         |
|                                                                         |
| def process_data(filename):                                             |
|                                                                         |
| df = pd.read_csv(filename, sep=\' \')                                   |
|                                                                         |
| data = df.values                                                        |
|                                                                         |
| x1 = data\[:, 1\]                                                       |
|                                                                         |
| x2 = data\[:, 2\]                                                       |
|                                                                         |
| y = data\[:, 3\]                                                        |
|                                                                         |
| with open(\'watermelon_3a_svm.txt\', \'w\') as f:                       |
|                                                                         |
| for m1, m2, n in zip(x1, x2, y):                                        |
|                                                                         |
| f.write(\"{} 1:{} 2:{}\\n\".format(int(n), round(m1, 3), round(m2, 3))) |
|                                                                         |
| f.close()                                                               |
|                                                                         |
| def main():                                                             |
|                                                                         |
| process_data(\'watermelon_3a.txt\')                                     |
|                                                                         |
| y, x = svm_read_problem(\'watermelon_3a_svm.txt\')                      |
|                                                                         |
| model1 = svm_train(y, x, \'-t 0 -c 1000\') \# 线性核                    |
|                                                                         |
| model2 = svm_train(y, x, \'-t 2 -c 1000 -g 0.8\') \# 高斯核             |
|                                                                         |
| svm_save_model(\'Linear.model\', model1)                                |
|                                                                         |
| svm_save_model(\'Gauss.model\', model2)                                 |
|                                                                         |
| svm_predict(y, x, model2)                                               |
|                                                                         |
| if \_\_name\_\_ == \'\_\_main\_\_\':                                    |
|                                                                         |
| main()                                                                  |
+-------------------------------------------------------------------------+

2、SVR.py

+------------------------------------------------------------+
| \# -\*- coding: utf-8 -\*                                  |
|                                                            |
| \# \@Time: 2021/5/13 13:20                                 |
|                                                            |
| import pandas as pd                                        |
|                                                            |
| from svmutil import \*                                     |
|                                                            |
| \# 将西瓜数据集3.0a转化为LIBSvm包规定的数据格式            |
|                                                            |
| def process_data(filename):                                |
|                                                            |
| df = pd.read_csv(filename, sep=\' \')                      |
|                                                            |
| data = df.values                                           |
|                                                            |
| x1 = data\[:, 1\]                                          |
|                                                            |
| y = data\[:, 2\]                                           |
|                                                            |
| with open(\'watermelon_3a_svr.txt\', \'w\') as f:          |
|                                                            |
| for m1, m2 in zip(x1, y):                                  |
|                                                            |
| f.write(\"{} 1:{}\\n\".format(round(m2, 3), round(m1, 3))) |
|                                                            |
| f.close()                                                  |
|                                                            |
| def main():                                                |
|                                                            |
| process_data(\'watermelon_3a.txt\')                        |
|                                                            |
| y, x = svm_read_problem(\'watermelon_3a_svm.txt\')         |
|                                                            |
| model3 = svm_train(y, x, \'-t 3 -c 100 -g 10\') \# 线性核  |
|                                                            |
| svm_save_model(\'SVR.model\', model3)                      |
|                                                            |
| svm_predict(y, x, model3)                                  |
|                                                            |
| if \_\_name\_\_ == \'\_\_main\_\_\':                       |
|                                                            |
| main()                                                     |
+------------------------------------------------------------+