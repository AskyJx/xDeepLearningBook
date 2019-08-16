"""
Created: May 2018
@author: JerryX
Find more at: https://www.zhihu.com/people/xu-jerry-82
nn with on FC layer
"""
import numpy as np
import logging.config
import random, time
import struct
import matplotlib.pyplot as plt

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(rootPath+'/xDLbase')
sys.path.append(rootPath+'/xutils')

from xDLbase.xview import *
from xDLbase.fc import *
from xDLbase.optimizers import *
from xDLbase.activators import *
from xDLbase.session import *


# create logger
exec_abs = os.getcwd()
log_conf = exec_abs + '/config/logging.conf'
logging.config.fileConfig(log_conf,disable_existing_loggers=0)
logger = logging.getLogger('main')
# 持久化配置
trace_file_path = 'D:/0tmp/'
exec_name = os.path.basename(__file__)
trace_file = trace_file_path + exec_name + ".data"

# 训练数据
path_minst_unpack = 'd:\\0tmp\\04_dataset\\MNIST\\MNIST_data_unpack'

# General params
class Params:
    # 持久化开关
    TRACE_FLAG = False
    # loss曲线开关
    SHOW_LOSS_CURVE = True
    INIT_W = 0.01  # 权值初始化参数
    LRT = 0.1

    EPS = 1e-8
    INIT_RNG=1e-4
    EPOCH_NUM = 9  # EPOCH
    MINI_BATCH_SIZE = 512  # batch_size
    ITERATION = 10  # 每batch训练轮数
    TYPE_K = 10  # 分类类别
    VALIDATION_CAPACITY = 2000  # 验证集大小
    VAL_FREQ = 10  # val per how many batches
    LOG_FREQ = 10000  # Log per how many batches
    IMAGE_SIZE = 28
    IMAGE_SIZE_L = 784

    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32

    FC1_SIZE_INPUT = 3136
    FC1_SIZE_OUTPUT = 512

# data loading
class MnistData(object):

    def __init__(self, absPath, is4Cnn, dataType):
        self.absPath = absPath
        self.is4Cnn = is4Cnn  # True for cnn,False for other nn structures
        self.dataType = dataType
        self.imgs, self.labels = self._load_mnist_data(kind='train')
        self.imgs_v, self.labels_v = self._load_mnist_data(kind='t10k')
        self.sample_range = [i for i in range(len(self.labels))]  # 训练样本范围
        self.sample_range_v = [i for i in range(len(self.labels_v))]  # 验证样本范围

    # 加载mnist
    def _load_mnist_data(self, kind='train'):
        labels_path = os.path.join(self.absPath, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(self.absPath, '%s-images.idx3-ubyte' % kind)

        with open(labels_path, 'rb') as labelfile:
            # 读取前8个bits
            magic, n = struct.unpack('>II', labelfile.read(8))
            # 余下的数据读到标签数组中
            labels = np.fromfile(labelfile, dtype=np.uint8)

        with open(images_path, 'rb') as imagefile:
            # 读取前16个bit
            magic, num, rows, cols = struct.unpack('>IIII', imagefile.read(16))
            # 余下数据读到image二维数组中，28*28=784像素的图片共60000张（和标签项数一致）
            # reshape 从原数组创建一个改变尺寸的新数组(28*28图片拉直为784*1的数组)
            # CNN处理的输入则reshape为28*28*1
            if False == self.is4Cnn:
                images_ori = np.fromfile(imagefile, dtype=np.uint8).reshape(len(labels), 784)
            else:
                # 支持多通道，此处通道为1
                images_ori = np.fromfile(imagefile, dtype=np.uint8).reshape(len(labels), 1, 28, 28)
            # 归一化
            images = images_ori / 255
        return images, labels

    # 对训练样本序号随机分组
    def getTrainRanges(self, miniBatchSize):

        rangeAll = self.sample_range
        random.shuffle(rangeAll)
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs

    # 获取训练样本范围对应的图像和标签
    def getTrainDataByRng(self, rng):

        xs = np.array([self.imgs[sample] for sample in rng], self.dataType)
        values = np.array([self.labels[sample] for sample in rng])
        return xs, values

    # 获取随机验证样本
    def getValData(self, valCapacity):

        samples_v = random.sample(self.sample_range_v, valCapacity)
        #  验证输入 N*28*28
        images_v = np.array([self.imgs_v[sample_v] for sample_v in samples_v], dtype=self.dataType)
        #  正确类别 1*K
        labels_v = np.array([self.labels_v[sample_v] for sample_v in samples_v])

        return images_v, labels_v

def main():
    logger.info('start..')
    # 初始化
    try:
        os.remove(trace_file)
    except FileNotFoundError:
        pass

    if (True == Params.SHOW_LOSS_CURVE):
        view = ResultView(Params.EPOCH_NUM,
                          ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
                          ['y', 'r', 'g', 'b'],
                          ['Iteration', 'Loss', 'Accuracy'],
                          Params.DTYPE_DEFAULT)
    # time stamp
    s_t = 0

    # 数据对象初始化
    mnist = MnistData(path_minst_unpack, False, Params.DTYPE_DEFAULT)

    optmParams = None  # SGD
    OptimizerCLS = SGDOptimizer


    # 定义网络结构，支持各层使用不同的优化方法。
    # 输入层->FC1->FC2->softmax->输出结果
    # 输入层->FC1->softmax->输出结果

    fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.IMAGE_SIZE_L, Params.FC1_SIZE_OUTPUT, ReLU,
                  OptimizerCLS, optmParams,False, Params.DTYPE_DEFAULT,Params.INIT_W)

    fc2 = FCLayer(Params.MINI_BATCH_SIZE, Params.FC1_SIZE_OUTPUT, Params.TYPE_K, NoAct,
                  OptimizerCLS, optmParams, False, Params.DTYPE_DEFAULT,Params.INIT_W)
    dnnLayers = [fc1,fc2]
    # 生成训练会话实例
    sess = Session(dnnLayers,SoftmaxCrossEntropyLoss)

    # 开始训练过程
    iter = 0
    for epoch in range(Params.EPOCH_NUM):

        lrt = Params.LRT
        logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 准备epoch随机训练样本
        dataRngs = mnist.getTrainRanges(Params.MINI_BATCH_SIZE)

        # 开始训练
        for batch in range(len(dataRngs)):
            start = time.time()
            x, y_ = mnist.getTrainDataByRng(dataRngs[batch])
            for iteration in range(Params.ITERATION):
                acc_t, loss_t = sess.train_steps(x, y_, lrt)
                iter += 1
            if (batch % Params.LOG_FREQ == 0):  # FREQ个batch show一次日志
                logger.info("epoch %2d-%3d, loss= %.8f,acc_t= %.3f st[%.1f]" % (
                    epoch, batch, loss_t, acc_t, s_t))

            # 使用随机验证样本验证结果
            if (batch % Params.VAL_FREQ == 0 and (batch+epoch) >0):
                x_v, y_v = mnist.getValData(Params.VALIDATION_CAPACITY)
                y, loss_v,acc_v = sess.validation(x_v, y_v)

                logger.info('epoch %2d-%3d, loss=%f, loss_v=%f, acc=%f, acc_v=%f' % (
                    epoch, batch, loss_t, loss_v, acc_t, acc_v))
                # 可视化记录
                if (True == Params.SHOW_LOSS_CURVE):
                    view.addData(iter,loss_t, loss_v, acc_t, acc_v)

            s_t = time.time() - start


    logger.info('session end')
    view.show()

if __name__ == '__main__':
    main()
