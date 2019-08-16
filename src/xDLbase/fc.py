"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools

#Tools = xnnUtils.Tools()
# 全连接类
class FCLayer(object):
    def __init__(self, miniBatchesSize, i_size, o_size,
                 activator, optimizerCls,optmParams,
                 needReshape,dataType,init_w):
        # 初始化超参数
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.o_size = o_size
        self.activator = activator
        self.optimizerObj = optimizerCls(optmParams, dataType)
        # 是否将N,T,D输入，先拉伸成N,T*D,再做仿射变换
        # 在letNet-5中，可以将pooling层输出3维拉成2维
        # 在RNN中，可以将N v M中的N个T时刻输出D维向量，
        # 变成N,T*D ,再仿射变换 为 N*D' 规格, 起到 N->M映射的效果
        self.needReshape = needReshape
        self.dataType = dataType
        self.w = init_w * np.random.randn(i_size, o_size).astype(dataType)
        self.b = np.zeros(o_size, dataType)
        self.out = []
        self.deltaPrev = []  # 上一层激活后的误差输出
        self.deltaOri = []  # 本层原始误差输出
        self.shapeOfOriIn = () # 原始输入维度
        self.inputReshaped =[]
    # 预测时前向传播
    def inference(self, input):
        self.shapeOfOriIn = input.shape
        self.out = self.fp(input)
        return self.out

    # 前向传播,激活后再输出
    def fp(self, input):
        # 拉伸变形处理
        self.shapeOfOriIn = input.shape
        self.inputReshaped = input if self.needReshape is False else input.reshape(input.shape[0],-1)
        self.out = self.activator.activate(Tools.matmul(self.inputReshaped, self.w) + self.b)
        ####debug####
        # np.savetxt('G:/0tmp/0debug/x.csv',self.inputReshaped[0])
        # np.savetxt('G:/0tmp/0debug/w_c1.csv', self.w[:,0])
        # np.savetxt('G:/0tmp/0debug/w_c2.csv', self.w[:, 1])
        # np.savetxt('G:/0tmp/0debug/out.csv', self.out[0])
        ####debug end#####
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta, lrt):
        self.deltaOri = self.activator.bp(delta, self.out)

        # # 恢复拉伸变形
        # self.deltaOri = deltaOri_reshaped if self.needReshape is False else deltaOri_reshaped.reshape(self.shapeOfOriIn)

        self.bpDelta()
        self.bpWeights(input, lrt)

        return self.deltaPrev

    # 输出误差反向传播至上一层
    def bpDelta(self):

        deltaPrevReshapped = Tools.matmul(self.deltaOri, self.w.T)

        self.deltaPrev = deltaPrevReshapped if self.needReshape is False else deltaPrevReshapped.reshape(self.shapeOfOriIn)
        return self.deltaPrev

    # 计算反向传播权重梯度w,b
    def bpWeights(self, input, lrt):
        # dw = Tools.matmul(input.T, self.deltaOri)
        dw = Tools.matmul(self.inputReshaped.T, self.deltaOri)
        db = np.sum(self.deltaOri, axis=0, keepdims=True).reshape(self.b.shape)
        weight = (self.w,self.b)
        dweight = (dw,db)
        # 元组按引用传递，值在方法内部已被更新
        self.optimizerObj.getUpdWeights(weight,dweight, lrt)