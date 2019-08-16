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
from lossfunc import *

# 会话类

class Session(object):
    # 构造各层
    def __init__(self, layers, lossfunCls):
        self.layers = layers
        self.input = []
        self.lossCls = lossfunCls

    # 前向传播和验证
    # val = False 训练 ， True 测试/预测
    def inference(self, train_data, y_, val=False):
        curr_batch_size = train_data.shape[0]
        self.input = train_data
        dataLayer = train_data

        # 训练
        if False == val:
            for layer in self.layers:
                dataLayer = layer.fp(dataLayer)

        else:  # 预测/测试
            for layer in self.layers:
                dataLayer = layer.inference(dataLayer)

        #y = dataLayer[:,-1,:]
        y = dataLayer
        data_loss,delta,acc = self.lossCls.loss(y,y_,curr_batch_size)


        return y, data_loss, delta, acc

    def bp(self, delta, lrt):
        deltaLayer = delta
        for i in reversed(range(1, len(self.layers))):
            deltaLayer = self.layers[i].bp(self.layers[i - 1].out, deltaLayer, lrt)

        # 第一层，以输入作为输出,误差不再反向传播
        # 对dh,只有最后一个T有梯度，其余时刻梯度为0
        self.layers[0].bp(self.input, deltaLayer, lrt)

    # 实现训练步骤
    def train_steps(self, train_data, y_, lrt):
        _,loss, delta,acc = self.inference(train_data, y_, val=False)
        self.bp(delta, lrt)
        return acc, loss

    # 独立数据集验证训练结果
    def validation(self, data_v, y_v):
        y, loss,_,acc = self.inference(data_v, y_v, val=True)
        return y, loss , acc