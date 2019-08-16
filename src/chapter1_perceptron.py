"""
Created: May 2018
@author: JerryX
Find more at: https://www.zhihu.com/people/xu-jerry-82
Perceptron
"""
import os

os.environ['PYGLET_SHADOW_WINDOW'] = "0"
import matplotlib.pyplot as plt

import numpy as np
import logging.config
from mpl_toolkits.mplot3d import axes3d

# create logger
exec_abs = os.getcwd()
log_conf = exec_abs + '/config/logging.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('main')

# 固定随机数种子
np.random.seed(0)

# 定义初始超平面参数，用于生成线性可分实例点
# W_ = np.array([0.1, 0.9, -1])
W_ = [2, -0.9, 1]
b_ = -1
LRT = 0.1

# 根据预定义超平面，补足线性可分实例点的缺失一维
def fun(w, b, x1, x2):
    x3 = -(w[0] * x1 + w[1] * x2 + b) / w[2]
    return x3

# 生成200个线性可分实例点，正负各半
class DataScatter(object):
    def __init__(self):
        self.w_ = W_
        self.b_ = b_
        self.x, self.y_ = self.createSpots()

    # 生成n个[vmin, vmax)之间的均匀分布样本返回array
    def randrange(self, n, vmin, vmax):
        return (vmax - vmin) * np.random.rand(n) + vmin

    def createSpots(self):
        n = 100
        ti = self.randrange(n, 0, 100)
        xs1 = self.randrange(n, 0, 100)
        ys1 = self.randrange(n, 0, 100)
        zs1 = fun(W_, b_, xs1, ys1) + ti  # 正实例点

        ti = self.randrange(n, 0, 100)
        xs2 = self.randrange(n, 0, 100)
        ys2 = self.randrange(n, 0, 100)
        zs2 = fun(W_, b_, xs2, ys2) - ti  # 负实例点

        x = np.zeros((2 * n, 3))
        y_ = np.zeros(2 * n)
        x[0:n, 0] = xs1
        x[0:n, 1] = ys1
        x[0:n, 2] = zs1
        y_[0:n] = 1  # 正确分类 +1

        x[n:2 * n, 0] = xs2
        x[n:2 * n, 1] = ys2
        x[n:2 * n, 2] = zs2
        y_[n:2 * n] = -1  # 正确分类 -1

        return x, y_


class Perceptron(object):

    # x:(200,3), w:(1,3)
    # 参数初始化为0
    def __init__(self):
        self.w = np.zeros((1, 3), np.float32)
        self.b = 0

    # 训练过程，得到满足要求的超平面参数
    def train(self, x, y_, lrt):

        iter = 0

        errExcists = True
        while (errExcists):
            errExcists = False
            i = 0
            for i in range(x.shape[0]):
                y = np.dot(self.w, x[i]) + self.b
                if y_[i] * y <= 0:  # 误分类点
                    self.w += lrt * y_[i] * x[i]
                    self.b += lrt * y_[i]
                    iter += 1
                    logger.info("-iter:%d w:[%f %f %f] b:%f" % (
                        iter, self.w[0, 0], self.w[0, 1], self.w[0, 2], self.b))
                    errExcists = True  # 存在误分类点,继续迭代
                    break


class view(object):

    # 返回由法向量和偏移确定的超平面离散坐标，用于绘图
    def getHypePlane(self, w, b):
        vec = np.arange(0, 100, 1)
        x, y = np.meshgrid(vec, vec)
        z = fun(w, b, x, y)

        # 返回超平面三元tuple
        return (x, y, z)

    def show(self, n, x, hp):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x[0:n, 0], x[0:n, 1], x[0:n, 2], c='b', marker='x')
        ax.scatter(x[n:2 * n, 0], x[n:2 * n, 1], x[n:2 * n, 2], c='r', marker='o')

        ax.plot_surface(hp[0], hp[1], hp[2], rstride=1, cstride=1, alpha=0.8, cmap=plt.cm.coolwarm)

        ax.set_xlabel('$x^{(1)}$ Feature', color='r')
        ax.set_ylabel('$x^{(2)}$  Feature', color='r')
        ax.set_zlabel('$x^{(3)}$  Feature', color='r')

        plt.show()


def main():
    data = DataScatter()
    logger.info("data created")
    model = Perceptron()
    model.train(data.x, data.y_, LRT)

    vw = view()
    hp_ = vw.getHypePlane(data.w_, data.b_)
    vw.show(100, data.x, hp_)

    hp = vw.getHypePlane(model.w[0], model.b)
    vw.show(100, data.x, hp)


if __name__ == '__main__':
    main()
