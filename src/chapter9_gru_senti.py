"""
Created: May 2018
@author: JerryX
Find more at: https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np
import logging.config
import random, time
import matplotlib.pyplot as plt
import gc

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

TRAIN_VEC='G:/0tmp/nlp_data/aclImdb/train/train_all_vec_400.npy'


# General params
class Params:

    EPOCH_NUM = 5  # EPOCH
    N_CLASS = 2 # 分类类别
    VERIFY_PER = 0.01 # 验证集占比
    MINI_BATCH_SIZE = 8  # batch_size
    ITERATION = 1  # 每batch训练轮数
    # LEARNING_RATE = 0.018
    # LEARNING_RATE = 0.003
    # LEARNING_RATE = 0.005
    DIC_L_RATE = {1: 0.01,2: 0.005, 3: 0.005, 4: 0.002, 5: 0.002, 6: 0.001, 7: 0.001, 8: 0.0005, 100: 0.0002}
    # DIC_L_RATE = {1: 0.02,2: 0.01, 3: 0.01, 4: 0.005, 5: 0.0025, 6: 0.0025, 7: 0.001, 8: 0.001, 100: 0.005}
    # DIC_L_RATE = {1: 0.05,2: 0.025, 3: 0.025, 4: 0.012, 5: 0.012, 6: 0.012, 7: 0.005, 8: 0.001, 100: 0.005}
    LRT_TIMES = 1
    VAL_FREQ = 100  # val per how many batches
    LOG_FREQ = 10  # log per how many batches

    DROPOUT_R_RATE = 0.8 # 保留率
    HIDDEN_SIZE = 16  # LSTM中隐藏节点的个数,每个时间节点上的隐藏节点的个数，是w的维度.
    # RNN/LSTM/GRU每个层次的的时间节点个数，有输入数据的元素个数确定。
    NUM_LAYERS = 3  # RNN/LSTM的层数。
    # 设置缺省数值类型
    DTYPE_DEFAULT = np.float32
    INIT_W = 0.01  # 权值初始化参数

    TIMESTEPS = 200 # 循环神经网络的训练序列长度。
    PRED_TYPE = 2  # 预测类别数

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
class SentiData(object):

    def __init__(self, absPath, dataType):
        self.dataType = dataType
        self.dataf = absPath
        # self.x, self.y, self.x_v, self.y_v = self.initData()
        # self.mu=0
        # self.var=0
        self.x, self.y, self.x_v,self.y_v = self.initData()

        self.sample_range = [i for i in range(len(self.y))]  # 训练样本范围
        # self.sample_range_v = [i for i in range(len(self.y_v))]  # 验证样本范围

    # 加载向量化的句子
    def _load_senti_data(self):

        train_all= np.load(TRAIN_VEC)

        train_all = train_all.astype(Params.DTYPE_DEFAULT)

        return train_all

    def initData(self):
        train_X, train_y, pred_X,pred_y = self.generate_data_debug(self._load_senti_data())

        return train_X, train_y, pred_X, pred_y

    ##产生数据。
    def generate_data(self,data_all):
        num_all = data_all.shape[0] # 总记录数
        num_verify= int(num_all*Params.VERIFY_PER) # 验证记录数

        #for half
        y_v =[1 for i in range(500)]
        y_v.extend([0 for i in range(500)])
        y=[1 for i in range(12000)]
        y.extend([0 for i in range(12000)])

        # # label for all
        # y_v =[1 for i in range(1000)]
        # y_v.extend([0 for i in range(1000)])
        # y=[1 for i in range(24000)]
        # y.extend([0 for i in range(24000)])

        # train_pos 1.2w
        X = data_all[0:12000]
        # train_pos 500 for validation
        X_V = data_all[12000:12500]

        # # test_pos 1.2w
        # X = np.concatenate([X, data_all[25000:37000]], axis=0)
        # # test_pos 500 for validation
        # X_V = np.concatenate([X_V, data_all[37000:37500]], axis=0)

        # train_neg 1.2w
        X = np.concatenate([X,data_all[12500:24500]],axis=0)
        # train_neg 500 for validation
        X_V = np.concatenate([X_V,data_all[24500:25000]],axis =0)

        # # test_neg 1.2w
        # X = np.concatenate([X, data_all[37500:49500]], axis=0)
        # # test_neg 500 for validation
        # X_V = np.concatenate([X_V, data_all[49500:]], axis=0)
        del data_all
        gc.collect()

        return X,y,X_V,y_v

    ##产生数据。
    def generate_data_debug(self,data_all):
        num_all = data_all.shape[0] # 总记录数取十分之一做测试
        num_verify= int(num_all*Params.VERIFY_PER) # 验证记录数

        #for half
        y_v =[1 for i in range(500)]
        y_v.extend([0 for i in range(500)])
        y=[1 for i in range(1000)]
        y.extend([0 for i in range(1000)])


        X = data_all[0:1000]
        # train_pos 500 for validation
        X_V = data_all[1000:1500]


        X = np.concatenate([X,data_all[1500:2500]],axis=0)

        X_V = np.concatenate([X_V,data_all[2500:3000]],axis =0)

        del data_all
        gc.collect()

        return X,y,X_V,y_v


    # 对训练样本序号随机分组
    def getTrainRanges(self, miniBatchSize):

        rangeAll = self.sample_range
        random.shuffle(rangeAll)
        rngs = [rangeAll[i:i + miniBatchSize] for i in range(0, len(rangeAll), miniBatchSize)]
        return rngs

    # 获取训练样本范围对应的样本和标签
    def getTrainDataByRng(self, rng):

        xs = np.array([self.x[sample] for sample in rng], self.dataType)
        values = np.array([self.y[sample] for sample in rng])
        return xs, values

    # 获取验证样本,不打乱，用于显示连续曲线
    def getValData(self):

        return self.x_v,self.y_v

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
                      ['y', 'r', 'g', 'b'],
                      ['Iteration', 'Loss', 'Accuracy'],
                      Params.DTYPE_DEFAULT)
    s_t = 0

    # 数据对象初始化
    seqData = SentiData(TRAIN_VEC,Params.DTYPE_DEFAULT)


    # 定义网络结构，支持各层使用不同的优化方法。

    optmParamsRnn1 = (Params.BETA1, Params.BETA2, Params.EPS)
    optimizer = AdamOptimizer

    # rnn
    # rnn1 = RnnLayer('rnn1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DTYPE_DEFAULT,Params.DROPOUT_R_RATE,Params.INIT_RNG)

    # LSTM
    # rnn1 = LSTMLayer('lstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,Params.NUM_LAYERS,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE, Params.DTYPE_DEFAULT,Params.INIT_RNG)
    # rnn1 = BiLSTMLayer('bilstm1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,Params.NUM_LAYERS,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE, Params.DTYPE_DEFAULT,Params.INIT_RNG)

    #GRU
    # rnn1 = GRULayer('gru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)
    rnn1 = BiGRULayer('bigru1',Params.MINI_BATCH_SIZE,Params.HIDDEN_SIZE,3,optimizer,optmParamsRnn1,Params.DROPOUT_R_RATE,Params.DTYPE_DEFAULT,Params.INIT_RNG)

    optmParamsFc1=(Params.BETA1, Params.BETA2, Params.EPS)
    # fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.TIMESTEPS*Params.HIDDEN_SIZE, Params.PRED_TYPE, NoAct,
    fc1 = FCLayer(Params.MINI_BATCH_SIZE, Params.TIMESTEPS*2* Params.HIDDEN_SIZE, Params.PRED_TYPE, NoAct,
    # fc1 = FCLayer(Params.MINI_BATCH_SIZE, 2*Params.HIDDEN_SIZE, Params.PRED_TYPE, NoAct,
                  AdamOptimizer, optmParamsFc1,True, Params.DTYPE_DEFAULT,Params.INIT_W)

    cnnLayers = [rnn1,fc1]


    # 生成训练会话实例
    # sess = Session(seqLayers,MseLoss)
    sess = Session(cnnLayers,SoftmaxCrossEntropyLoss)
    # 开始训练过程
    iter = 0
    for epoch in range(Params.EPOCH_NUM):
        # 获取当前epoch使用的learing rate
        for key in Params.DIC_L_RATE.keys():
            if (epoch + 1) < key:
                break
            lrt = Params.DIC_L_RATE[key]
        # lrt = Params.LEARNING_RATE
        lrt = lrt * Params.LRT_TIMES
        logger.info("epoch %2d, learning_rate= %.8f" % (epoch, lrt))
        # 准备epoch随机训练样本
        dataRngs = seqData.getTrainRanges(Params.MINI_BATCH_SIZE)

        # 开始训练
        for batch in range(len(dataRngs)):
            start = time.time()
            x, y = seqData.getTrainDataByRng(dataRngs[batch])
            acc, loss_t = sess.train_steps(x, y, lrt)
            del x
            gc.collect()
            iter += 1

            if (batch % Params.LOG_FREQ == 0):  # 若干个batch show一次日志
                logger.info("epoch %2d-%3d, acc=%f loss= %.8f st[%.1f]" % (
                    epoch, batch,acc, loss_t,  s_t))

            # 使用随机验证样本验证结果
            # if (batch % Params.VAL_FREQ == 0 and (batch + epoch) > 0):
            if (batch % Params.VAL_FREQ == 0 and batch != 0):
                x_v, y_v = seqData.getValData()
                y, loss_v,acc_v = sess.validation(x_v, y_v)

                logger.info('epoch %2d-%3d, acc=%f,loss=%.8f, acc_v=%f loss_v=%.8f ' % (
                    epoch, batch, acc,loss_t, acc_v,loss_v))

                if (True == Params.SHOW_LOSS_CURVE):
                    # view.addData(fc1.optimizerObj.Iter,
                    view.addData(iter,
                                 loss_t, loss_v, acc, acc_v)
            s_t = time.time() - start

    logger.info('session end')
    x_v, y_v = seqData.getValData()
    y, loss_v,acc_v = sess.validation(x_v, y_v)
    logger.info('epoch %2d-%3d, acc_v=%f loss_v=%.8f ' % (
        epoch, batch,  acc_v, loss_v))


    logger.info('session end')
    view.show()

if __name__ == '__main__':
    main_rnn()
