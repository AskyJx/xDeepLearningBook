"""
Created: May 2018
@author: JerryX
Find more at: https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np
import logging.config
import random, time
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
from xDLbase.rnn import *
from xDLbase.optimizers import *
from xDLbase.activators import *
from xDLbase.session import *

# create logger
exec_abs = os.getcwd()
log_conf = exec_abs + '/config/logging.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('main')

# 持久化配置
trace_file_path = 'D:/0tmp/'
exec_name = os.path.basename(__file__)
trace_file = trace_file_path + exec_name + ".data"

# General params
class Params:

    EPOCH_NUM = 5  # EPOCH
    MINI_BATCH_SIZE = 32  # batch_size
    ITERATION = 1  # 每batch训练轮数
    LEARNING_RATE = 0.005  # Vanilla E5:loss 0.0014, 好于AdaDelta的0.0021
    # LEARNING_RATE = 0.002  # LSTM
    # LEARNING_RATE = 0.002  # GRU
    # LEARNING_RATE = 0.1  # BiLSTM
    # LEARNING_RATE = 0.1  # BiGRU
    # LEARNING_RATE = 0.05  # BiGRU+ReLU
    # VAL_FREQ = 30  # val per how many batches
    VAL_FREQ = 50  # val per how many batches
    # LOG_FREQ = 10  # log per how many batches
    LOG_FREQ = 1000000  # log per how many batches


    HIDDEN_SIZE = 30  # LSTM中隐藏节点的个数,每个时间节点上的隐藏节点的个数，是w的维度.
    # RNN/LSTM/GRU每个层次的的时间节点个数，有输入数据的元素个数确定。
    NUM_LAYERS = 3  # RNN/LSTM的层数。
    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32
    INIT_W = 0.01  # 权值初始化参数

    DROPOUT_R_RATE = 1 # 保留率
    TIMESTEPS = 10  # 循环神经网络的训练序列长度。
    PRED_STEPS = TIMESTEPS  # 预测序列长度
    TRAINING_STEPS = 10000  # 训练轮数。
    TRAINING_EXAMPLES = 10000  # 训练数据个数。
    TESTING_EXAMPLES = 1000  # 测试数据个数。
    SAMPLE_GAP = 0.01  # 采样间隔。
    VALIDATION_CAPACITY = TESTING_EXAMPLES-TIMESTEPS  # 验证集大小

    # 持久化开关
    TRACE_FLAG = False
    # loss曲线开关
    SHOW_LOSS_CURVE = True

    # Optimizer params
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8
    EPS2 = 1e-10
    REG_PARA = 0.5  # 正则化乘数
    LAMDA = 1e-4  # 正则化系数lamda
    INIT_RNG=1e-4

    # 并行度
    # TASK_NUM_MAX = 3
    # 任务池
    # g_pool = ProcessPoolExecutor(max_workers=TASK_NUM_MAX)


# data loading
class SeqData(object):

    def __init__(self, dataType):
        self.dataType = dataType
        self.x, self.y,self.x_v, self.y_v = self.initData()

        self.sample_range = [i for i in range(len(self.y))]  # 训练样本范围
        self.sample_range_v = [i for i in range(len(self.y_v))]  # 验证样本范围

    def initData(self):
        # 用正弦函数生成训练和测试数据集合。
        # (1w+1k)*0.01
        test_start = (Params.TRAINING_EXAMPLES + Params.TIMESTEPS) * Params.SAMPLE_GAP
        #  (1w+1k)*0.01 + (1w+1k)*0.01
        test_end = test_start + (Params.TESTING_EXAMPLES + Params.TIMESTEPS) * Params.SAMPLE_GAP

        # np.linspace 生成等差数列(start-首项,stop-尾项,number-项数,endpoint-末项是否包含在内，默认true包含)
        # curve 内调用函数，从等差数列生成函数值序列
        train_X, train_y = self.generate_data(self.curve(np.linspace(
            0, test_start, Params.TRAINING_EXAMPLES + Params.TIMESTEPS, dtype=self.dataType)))
        test_X, test_y = self.generate_data(self.curve(np.linspace(
            test_start, test_end, Params.TESTING_EXAMPLES + Params.TIMESTEPS, dtype=self.dataType)))

        return train_X, train_y,test_X,test_y

    ##产生数据。
    def generate_data(self,seq):
        X = []
        y = []
        # 序列的第i项和后面的TIMESTEPS-1项合在一起的10个点序列作为输入；第i + TIMESTEPS项这个点作为输出。
        # 即用curve函数前面从i开始的TIMESTEPS=10个点的信息，预测第i + TIMESTEPS这个点的函数值。
        # 一共生成TRAINING_EXAMPLES = 1w对 序列、值数据对用于训练，同理生成验证数据
        # 交换维度为N,T,D :(N,10,1)->(N,1,10)
        # for i in range(len(seq) - Params.TIMESTEPS):
        for i in range(len(seq) - Params.TIMESTEPS-Params.PRED_STEPS):
            X.append([seq[i: i + Params.TIMESTEPS]])
            # y.append([seq[i + Params.TIMESTEPS]])
            y.append([seq[i + Params.TIMESTEPS:i + Params.TIMESTEPS + Params.PRED_STEPS]])
        # return np.swapaxes(np.array(X, dtype=self.dataType),1,2), np.array(y, dtype=self.dataType)
        return np.swapaxes(np.array(X, dtype=self.dataType),1,2), np.swapaxes(np.array(y, dtype=self.dataType),1,2)


    def curve(self,x):
        # return np.sin(np.pi * x / 50) + np.cos(np.pi * x / 50) + np.sin(np.pi * x / 25)
        return np.sin(np.pi * x / 3.) + np.cos(np.pi * x / 3.) + np.sin(np.pi * x / 1.5)++ np.random.uniform(-0.05,0.05,len(x))

    # 对训练样本序号随机分组
    def getTrainRanges(self, miniBatchSize):

        rangeAll = self.sample_range
        random.shuffle(rangeAll)
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs

    # 获取训练样本范围对应的图像和标签
    def getTrainDataByRng(self, rng):

        xs = np.array([self.x[sample] for sample in rng], self.dataType)
        values = np.array([self.y[sample] for sample in rng])
        return xs, values

    # 获取验证样本,不打乱，用于显示连续曲线
    def getValData(self, valCapacity):

        samples_v = [i for i in range(valCapacity)]
        #  验证输入 N*28*28
        x_v = np.array([self.x_v[sample_v] for sample_v in samples_v], dtype=self.dataType)
        #  正确类别 1*K
        y_v = np.array([self.y_v[sample_v] for sample_v in samples_v])

        return x_v, y_v

def main_rnn():
    logger.info('start..')
    # 初始化
    try:
        os.remove(trace_file)
    except FileNotFoundError:
        pass

    # if (True == Params.SHOW_LOSS_CURVE):
    view = ResultView(Params.EPOCH_NUM,
                      ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
                      ['k', 'r', 'g', 'b'],
                      ['Iteration', 'Loss', 'Accuracy'],
                      Params.DTYPE_DEFAULT)
    s_t = 0

    # 数据对象初始化
    seqData = SeqData(Params.DTYPE_DEFAULT)


    # 定义网络结构，支持各层使用不同的优化方法。

    # optmParamsRnn1 =  (Params.BETA1, Params.BETA2, Params.EPS)
    # optimizer = AdagradOptimizer

    # optmParamsRnn1 = (Params.BETA1,Params.EPS)
    # optimizer = AdaDeltaOptimizer

    optmParamsRnn1 = (Params.BETA1, Params.BETA2, Params.EPS)
    optimizer = AdamOptimizer

    # rnn
    rnn1 = RnnLayer('rnn1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,Params.NUM_LAYERS,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #LSTM
    # rnn1 = LSTMLayer('lstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #BiLSTM
    # rnn1 = BiLSTMLayer('Bilstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #GRU
    #rnn1 = GRULayer('gru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #BiGRU
    # rnn1 = BiGRULayer('bigru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)


    optmParamsFc1=(Params.BETA1, Params.BETA2, Params.EPS)
    # RNN输出全部T个节点，FC层先把B,T,H拉伸成N,T*H, 再用仿射变换的W T*H,D 得到 N*D输出。
    fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.TIMESTEPS*Params.HIDDEN_SIZE, Params.PRED_STEPS, NoAct,
                  AdamOptimizer, optmParamsFc1,True,Params.DTYPE_DEFAULT,Params.INIT_W)

    seqLayers = [rnn1,fc1]

    # 生成训练会话实例
    sess = Session(seqLayers,MseLoss)

    # 开始训练过程
    iter = 0
    for epoch in range(Params.EPOCH_NUM):
        # 获取当前epoch使用的learing rate
        # for key in Params.DIC_L_RATE.keys():
        #     if (epoch + 1) < key:
        #         break
        #     lrt = Params.DIC_L_RATE[key]
        lrt = Params.LEARNING_RATE
        # logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 准备epoch随机训练样本
        dataRngs = seqData.getTrainRanges(Params.MINI_BATCH_SIZE)

        # 开始训练
        for batch in range(len(dataRngs)):
            start = time.time()
            x, y_ = seqData.getTrainDataByRng(dataRngs[batch])
            # 输出序列，只需要比较第0维
            _, loss_t = sess.train_steps(x, y_[:,:,0], lrt)
            iter += 1


            if (batch % Params.LOG_FREQ == 0):  # 若干个batch show一次日志
                logger.info("epoch %2d-%3d, loss= %.8f st[%.1f]" % (
                    epoch, batch, loss_t,  s_t))

            # 使用随机验证样本验证结果
            if (batch % Params.VAL_FREQ == 0 and (batch + epoch) > 0):
                x_v, y_v = seqData.getValData(Params.VALIDATION_CAPACITY)
                # 多个出数值的序列，只需要比较0维
                y, loss_v,_ = sess.validation(x_v, y_v[:,:,0])

                logger.info('epoch %2d-%3d, loss=%f, loss_v=%f' % (
                    epoch, batch, loss_t, loss_v))

                if (True == Params.SHOW_LOSS_CURVE):
                    # view.addData(fc1.optimizerObj.Iter,
                    view.addData(iter,
                                 loss_t, loss_v, 0, 0)
            s_t = time.time() - start

    logger.info('session end')
    x_v, y_v = seqData.getValData(Params.VALIDATION_CAPACITY)
    y, loss_v,_ = sess.validation(x_v, y_v[:,:,0])
    plt.figure()
    plt.plot(y[:,0],linewidth=1.5, ls=':',label='predictions')
    plt.plot(y_v[:,0,0],linewidth=0.5, ls='-', label='real_curve')
    plt.legend()
    plt.show()
    view.show()

if __name__ == '__main__':
    main_rnn()
