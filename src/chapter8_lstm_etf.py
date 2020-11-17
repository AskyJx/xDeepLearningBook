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

# Data

etf_file = os.path.join("F:/cwork/Project/TF_py3.6/hs300/data/hs300_data_seq_nodate.csv")

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

    EPOCH_NUM = 20  # EPOCH
    MINI_BATCH_SIZE = 32  # batch_size
    ITERATION = 1  # 每batch训练轮数
    # LEARNING_RATE = 0.018
    LEARNING_RATE = 0.0015
    VAL_FREQ = 100  # val per how many batches
    LOG_FREQ = 10  # log per how many batches

    DROPOUT_R_RATE = 0.5  # 保留率
    HIDDEN_SIZE = 64  # LSTM中隐藏节点的个数,每个时间节点上的隐藏节点的个数，是w的维度.
    # RNN/LSTM/GRU每个层次的的时间节点个数，有输入数据的元素个数确定。
    NUM_LAYERS = 3  # RNN/LSTM的层数。
    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32
    INIT_W = 0.01  # 权值初始化参数

    TIMESTEPS = 64 # 循环神经网络的训练序列长度。
    PRED_STEPS = 20  # 预测序列长度

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

    DROPOUT_RATE = 0.5  # dropout%

    # 并行度
    # TASK_NUM_MAX = 3
    # 任务池
    # g_pool = ProcessPoolExecutor(max_workers=TASK_NUM_MAX)


# data loading
class EtfData(object):

    def __init__(self, absPath, dataType):
        self.dataType = dataType
        self.data = absPath
        # self.x, self.y, self.x_v, self.y_v = self.initData()
        self.mu=0
        self.var=0
        self.x, self.y, self.pred_x = self.initData()

        self.sample_range = [i for i in range(len(self.y))]  # 训练样本范围
        # self.sample_range_v = [i for i in range(len(self.y_v))]  # 验证样本范围

    # 加载mnist
    def _load_eft_data(self):

        seq = np.loadtxt(open(self.data, "rb"), dtype=float, delimiter=",", usecols=(1,2,3), skiprows=0)

            # 归一化
        self.mu = np.mean(seq, axis=0)
        xmu = seq - self.mu
        self.var = np.mean(xmu ** 2, axis=0)

        seq = (seq - self.mu)/self.var

        return seq

    def initData(self):
        train_X, train_y, pred_X = self.generate_data(self._load_eft_data())

        return train_X, train_y, pred_X

    ##产生数据。
    def generate_data(self,seq):
        X = []
        y = []
        for i in range(len(seq) - Params.TIMESTEPS-Params.PRED_STEPS):
            X.append(seq[i: i + Params.TIMESTEPS])
            y.append(seq[i + Params.TIMESTEPS:i + Params.TIMESTEPS + Params.PRED_STEPS])

        pred_X=[]
        pred_X.append(seq[len(seq)-Params.TIMESTEPS:])
        return np.array(X, dtype=self.dataType), np.array(y, dtype=self.dataType) , np.array(pred_X, dtype=self.dataType)


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
    def getValData(self):

        return self.x,self.y

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
    seqData = EtfData(etf_file,Params.DTYPE_DEFAULT)


    # 定义网络结构，支持各层使用不同的优化方法。

    # optmParamsRnn1 = (Params.BETA1,Params.EPS)
    # optimizer = AdaDeltaOptimizer

    optmParamsRnn1 = (Params.BETA1, Params.BETA2, Params.EPS)
    optimizer = AdamOptimizer

    # rnn
    #rnn1 = RnnLayer('rnn1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DTYPE_DEFAULT,Params.DROPOUT_R_RATE,Params.INIT_RNG)

    #LSTM
    rnn1 = LSTMLayer('lstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,Params.NUM_LAYERS,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #GRU
    # rnn1 = GRULayer('gru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE, Params.DTYPE_DEFAULT,Params.INIT_RNG)

    optmParamsFc1=(Params.BETA1, Params.BETA2, Params.EPS)
    fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.TIMESTEPS*Params.HIDDEN_SIZE, Params.PRED_STEPS, NoAct,
                  AdamOptimizer, optmParamsFc1,True, Params.DTYPE_DEFAULT,Params.INIT_W)

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
        logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 准备epoch随机训练样本
        dataRngs = seqData.getTrainRanges(Params.MINI_BATCH_SIZE)

        # 开始训练
        for batch in range(len(dataRngs)):
            start = time.time()
            x, y_ = seqData.getTrainDataByRng(dataRngs[batch])
            _, loss_t = sess.train_steps(x, y_[:,:,0], lrt)
            iter += 1

            if (batch % Params.LOG_FREQ == 0):  # 若干个batch show一次日志
                logger.info("epoch %2d-%3d, loss= %.8f st[%.1f]" % (
                    epoch, batch, loss_t,  s_t))

            # 使用随机验证样本验证结果
            # if (batch % Params.VAL_FREQ == 0 and (batch + epoch) > 0):
            if (batch % Params.VAL_FREQ == 0 and batch != 0):
                x_v, y_v = seqData.getValData()
                y, loss_v,_ = sess.validation(x_v, y_v[:,:,0])

                logger.info('epoch %2d-%3d, loss=%.8f, loss_v=%.8f' % (
                    epoch, batch, loss_t, loss_v))

                if (True == Params.SHOW_LOSS_CURVE):
                    # view.addData(fc1.optimizerObj.Iter,
                    view.addData(iter,
                                 loss_t, loss_v, 0, 0)
            s_t = time.time() - start

    logger.info('session end')
    x_v, y_v = seqData.getValData()
    y, loss_v,_ = sess.validation(x_v, y_v[:,:,0])

    pred_y, _, _ = sess.validation(seqData.pred_x,0.)

    plt.figure()

    #原始数据长[1-3442]，
    #训练数据长[1-3432] ,迭代用前20步预测后10步，
    #测试时，同样用[1-3432] 数据做测试时
    #预测时，用[3423~3442] 预测 [3443-3452]的数据
    #横坐标应该是3442+Pred_steps这么长
    len_oriData = x_v.shape[0] + Params.TIMESTEPS+ Params.PRED_STEPS
    h_ord = [i for i in range(1,len_oriData+Params.PRED_STEPS+1)]
    plt.axvline(len_oriData, linestyle="dotted",  marker='.',linewidth=1, color='r')  # 在x=3441这个位置画一条辅助线分隔历史和未来

    plt.plot(h_ord[Params.TIMESTEPS:len_oriData],np.hstack([y[:,0],y[-1,:]]),linewidth=1.5, ls=':', label='eval_curve')
    plt.plot(h_ord[Params.TIMESTEPS:len_oriData],np.hstack([y_v[:,0,0],y_v[-1,:,0]]),linewidth=0.5, ls='-', label='real_curve')
    plt.plot(h_ord[len_oriData:],pred_y[0,:], label='predictions',color="r")
    # plt.plot(y[:,0],linewidth=1.5, ls=':',label='predictions')
    # plt.plot(y_v[:,0,0],linewidth=0.5, ls='-', label='real_curve')
    plt.legend()
    plt.show()
    view.show()

if __name__ == '__main__':
    main_rnn()
