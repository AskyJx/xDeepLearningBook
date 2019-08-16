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

class MseLoss:

    @staticmethod
    def loss(y,y_, n):
        corect_logprobs = Tools.mse(y, y_)
        data_loss = np.sum(corect_logprobs) / n
        delta = (y - y_) / n

        return data_loss, delta ,None

class SoftmaxCrossEntropyLoss:
    @staticmethod
    def loss(y,y_, n):
        y_argmax = np.argmax(y, axis=1)
        softmax_y = Tools.softmax(y)
        acc = np.mean(y_argmax == y_)
        # loss
        # corect_logprobs = Tools.crossEntropyLogit(softmax_y, y_)
        # data_loss = np.sum(corect_logprobs) / n
        data_loss = Tools.crossEntropy(softmax_y, y_)
        # delta
        softmax_y[range(n), y_] -= 1
        delta = softmax_y / n

        return data_loss, delta, acc
