"""
Created: May 2018
@author: JerryX
Find more at: https://www.zhihu.com/people/xu-jerry-82
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
from xDLbase.cnn import *
from xDLbase.optimizers import *
from xDLbase.activators import *
from xDLbase.session import *
from xDLbase.bn import *

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
path_minst_unpack = 'D:\data\MNIST\MNIST_data_unpack'

# General params
class Params:
    # 持久化开关
    TRACE_FLAG = False
    # loss曲线开关
    SHOW_LOSS_CURVE = True
    INIT_W = 0.01  # 权值初始化参数
    DIC_L_RATE = {1: 0.002, 2: 0.001, 3: 0.0004, 4: 0.0002, 5: 0.0001, 6: 0.00002, 7: 0.00001, 100: 0.000004}


    # Adam params
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    EPS2 = 1e-10
    REG_PARA = 0.5  # 正则化乘数
    LAMDA = 1e-4  # 正则化系数lamda
    INIT_RNG=1e-4
    EPOCH_NUM = 5  # EPOCH
    MINI_BATCH_SIZE = 32  # batch_size
    ITERATION = 1  # 每batch训练轮数
    TYPE_K = 10  # 分类类别
    DROPOUT_RATE = 0.5  # dropout%
    VALIDATION_CAPACITY = 2000  # 验证集大小
    VAL_FREQ = 30  # val per how many batches
    LOG_FREQ = 30  # Log per how many batches
    IMAGE_SIZE = 28
    IMAGE_CHANNEL = 1  # MINST通道数为，可支持多通道

    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32

    # Hyper params
    CONV1_F_SIZE = 5
    CONV1_STRIDES = 1
    CONV1_O_SIZE = 28
    CONV1_O_DEPTH = 32

    POOL1_F_SIZE = 2
    POOL1_STRIDES = 2

    CONV2_F_SIZE = 5
    CONV2_STRIDES = 1
    CONV2_O_SIZE = 14
    CONV2_O_DEPTH = 64

    POOL2_F_SIZE = 2
    POOL2_STRIDES = 2

    FC1_SIZE_INPUT = 3136
    FC1_SIZE_OUTPUT = 512

    # 并行度
    # TASK_NUM_MAX = 3
    # 任务池
    # g_pool = ProcessPoolExecutor(max_workers=TASK_NUM_MAX)


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
    mnist = MnistData(path_minst_unpack, True, Params.DTYPE_DEFAULT)

    optmParamsAdam=(Params.BETA1, Params.BETA2, Params.EPS)
    # 定义网络结构，支持各层使用不同的优化方法。
    # 输入层->卷积层1->BN1->池化层1->卷积层2->BN2->池化层2->FC1->FC2->softmax->输出结果

    # 在Conv1和relu/pool1之间加入bn1,所以conv1的输出不再激活
    conv1 = ConvLayer('conv1', Params.MINI_BATCH_SIZE, Params.IMAGE_SIZE, Params.IMAGE_CHANNEL,
                      Params.CONV1_F_SIZE, Params.CONV1_O_DEPTH,
                      Params.CONV1_O_SIZE, Params.CONV1_STRIDES,
                      NoAct, AdamOptimizer,optmParamsAdam, Params.DTYPE_DEFAULT,Params.INIT_W)
    #在BN后再激活
    # bn1 = BNLayer('bn1',Params.EPS,Params.MINI_BATCH_SIZE,Params.CONV1_O_DEPTH,
    #                   Params.CONV1_O_SIZE,ReLU,AdamOptimizer,optmParamsAdam, Params.DTYPE_DEFAULT)
    bn1 = BNLayer('bn1',Params.EPS,Params.MINI_BATCH_SIZE,Params.CONV1_O_DEPTH,
                      Params.CONV1_O_SIZE,ReLU,AdamOptimizer,optmParamsAdam, Params.DTYPE_DEFAULT)

    pool1 = MaxPoolLayer('pool1', Params.MINI_BATCH_SIZE, Params.POOL1_F_SIZE,
                         Params.POOL1_STRIDES, False, Params.DTYPE_DEFAULT)

    # 在Conv2和relu/pool2之间加入bn2,conv2的输出不再激活
    conv2 = ConvLayer('conv2', Params.MINI_BATCH_SIZE, Params.CONV2_O_SIZE, Params.CONV1_O_DEPTH,
                      Params.CONV2_F_SIZE, Params.CONV2_O_DEPTH,
                      Params.CONV2_O_SIZE, Params.CONV2_STRIDES,
                      NoAct, AdamOptimizer,optmParamsAdam, Params.DTYPE_DEFAULT,Params.INIT_W)
    # bn2 = BNLayer('bn2',Params.EPS,Params.MINI_BATCH_SIZE,Params.CONV2_O_DEPTH,
    #                   Params.CONV2_O_SIZE,ReLU,AdamOptimizer,optmParamsAdam, Params.DTYPE_DEFAULT)
    bn2 = BNLayer('bn2',Params.EPS,Params.MINI_BATCH_SIZE,Params.CONV2_O_DEPTH,
                      Params.CONV2_O_SIZE,ReLU,AdamOptimizer,optmParamsAdam, Params.DTYPE_DEFAULT)

    pool2 = MaxPoolLayer('pool2', Params.MINI_BATCH_SIZE, Params.POOL2_F_SIZE,
                         Params.POOL2_STRIDES, True, Params.DTYPE_DEFAULT)

    fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.FC1_SIZE_INPUT, Params.FC1_SIZE_OUTPUT, ReLU,
                  AdamOptimizer, optmParamsAdam,False, Params.DTYPE_DEFAULT,Params.INIT_W)

    fc2 = FCLayer(Params.MINI_BATCH_SIZE, Params.FC1_SIZE_OUTPUT, Params.TYPE_K, NoAct,
                  AdamOptimizer, optmParamsAdam,False, Params.DTYPE_DEFAULT,Params.INIT_W)

    # With BN
    cnnLayers = [conv1, bn1, pool1, conv2,bn2, pool2, fc1, fc2]
    # 生成训练会话实例
    sess = Session(cnnLayers,SoftmaxCrossEntropyLoss)

    # 开始训练过程
    iter = 0
    for epoch in range(Params.EPOCH_NUM):
        # 获取当前epoch使用的learing rate
        for key in Params.DIC_L_RATE.keys():
            if (epoch + 1) < key:
                break
            lrt = Params.DIC_L_RATE[key]

        logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 准备epoch随机训练样本
        dataRngs = mnist.getTrainRanges(Params.MINI_BATCH_SIZE)

        # 开始训练
        for batch in range(len(dataRngs)):
            start = time.time()
            x, y_ = mnist.getTrainDataByRng(dataRngs[batch])
            acc_t, loss_t = sess.train_steps(x, y_, lrt)
            iter += 1
            if (batch % 10 == 0):  # 10个batch show一次日志
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

def bn_test():
    x = np.array(np.random.randint(-10, 10, 24)).reshape((2, 3, 2, 2))
    optmParamsAdam=(Params.BETA1, Params.BETA2, Params.EPS)
    bn1 = BNLayer('bn1',Params.EPS,2,3,
                      2,ReLU,AdamOptimizer,optmParamsAdam, Params.DTYPE_DEFAULT)
    out,cache = bn1.bnForward_tr(x,Params.BETA1, Params.BETA2, Params.EPS)

    # beta = np.zeros((3, 2, 2), Params.DTYPE_DEFAULT)
    # gamma = np.ones((3, 2, 2), Params.DTYPE_DEFAULT)
    out1,cache1 = bn1.bnForward_inf(x, Params.BETA1, Params.BETA2, Params.EPS)

    pass

if __name__ == '__main__':
    # bn_test()
    main()

