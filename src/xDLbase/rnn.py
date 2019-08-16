"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np
import operator as op
import numba

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools
from activators import ReLU

# Rnn类
class RnnLayer(object):
    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType,  init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志
        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]

        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.rnnParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        rnnParams = []
        for layer in range(layersNum):

            Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            b = np.zeros(H, dataType)
            rnnParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})

        self.rnnParams = rnnParams

    def _initNnWeightOrthogonal(self, D, H, layersNum, dataType):

        # 层次
        rnnParams = []
        for layer in range(layersNum):

            # Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            Wh = Tools.initOrthogonal( (H, H),self.init_rng, dataType)

            DH = D if 0 == layer else H
            Wx = Tools.initOrthogonal( (DH, H),self.init_rng, dataType)
            b = np.zeros(H, dataType)
            rnnParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})

        self.rnnParams = rnnParams


    # 训练时前向传播
    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)
        return self.out

    # 预测时前向传播,激活后再输出
    # input: batch x seqNum, 32*10
    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            #self._initNnWeight(D, H, L, self.dataType)
            self._initNnWeightOrthogonal(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.rnn_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        # 全部输出,未用到的部分梯度为0
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    # TODO 实现反向传播逻辑，先按照时间，再按照层次，再更新Wx/Wf/b/V/bv 及偏置的反向传播梯度
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        N, T, D = input.shape
        H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.rnn_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        # for l in range(L - 1, -1, -1):
        for l in range(L):
            w = (self.rnnParams[l]['Wx'], self.rnnParams[l]['Wh'], self.rnnParams[l]['b'])
            # 此处不赋值也可以，因为是按引用传参
            # self.rnnParams[l]['Wx'], self.rnnParams[l]['Wh'], self.rnnParams[l]['b'] = self.optimizerObjs[l].getUpdWeights(w,dw[L-1-l],lrt)
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def rnn_forward(self, x):
        """
        Run a vanilla RNN forward on an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The RNN uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the RNN forward, we return the hidden states for all timesteps.

        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D).
        - h0: Initial hidden state, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - h: Hidden states for the entire timeseries, of shape (N, T, H).
        - cache: Values needed in the backward pass
        """

        h, cache = None, None
        ##############################################################################
        # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
        # input data. You should use the rnn_step_forward function that you defined  #
        # above. You can use a for loop to help compute the forward pass.            #
        ##############################################################################

        N, T, D = x.shape
        L = self.layersNum
        H = self.rnnParams[0]['b'].shape[0]
        xh = x
        for layer in range(L):

            h = np.zeros((N, T, H))
            h0 = np.zeros((N, H))
            cache = []
            for t in range(T):
                h[:, t, :], tmp_cache = self.rnn_step_forward(xh[:, t, :],
                                                              h[:, t - 1, :] if t > 0 else h0,
                                                              self.rnnParams[layer]['Wx'], self.rnnParams[layer]['Wh'],
                                                              self.rnnParams[layer]['b'])
                cache.append(tmp_cache)
            xh = h  # 之后以h作为xh作为跨层输入
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            self.rnnParams[layer]['h'] = h
            self.rnnParams[layer]['cache'] = cache

        return h  # 返回最后一层作为输出

    def rnn_backward(self, dh):
        """
        Compute the backward pass for a vanilla RNN over an entire sequence of data.

        Inputs:
        - dh: Upstream gradients of all hidden states, of shape (N, T, H).

        NOTE: 'dh' contains the upstream gradients produced by the
        individual loss functions at each timestep, *not* the gradients
        being passed between timesteps (which you'll have to compute yourself
        by calling rnn_step_backward in a loop).

        Returns a tuple of:
        - dx: Gradient of inputs, of shape (N, T, D)
        - dh0: Gradient of initial hidden state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        - db: Gradient of biases, of shape (H,)
        """
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        ##############################################################################
        # TODO: Implement the backward pass for a vanilla RNN running an entire      #
        # sequence of data. You should use the rnn_step_backward function that you   #
        # defined above. You can use a for loop to help compute the backward pass.   #
        ##############################################################################
        N, T, H = dh.shape
        x, _, _, _, _ = self.rnnParams[0]['cache'][0]
        D = x.shape[1]

        # 初始化最上一层误差

        dh_prevl = dh
        # 保存各层dwh,dwx,和db
        dweights = []
        # 逐层倒推
        for layer in range(self.layersNum - 1, -1, -1):
            # 得到前向传播保存的cache数组
            cache = self.rnnParams[layer]['cache']

            DH = D if layer == 0 else H
            dx = np.zeros((N, T, DH))
            dWx = np.zeros((DH, H))
            dWh = np.zeros((H, H))
            db = np.zeros(H)
            dprev_h_t = np.zeros((N, H))
            # 倒序遍历
            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h_t, dWx_t, dWh_t, db_t = self.rnn_step_backward(dh_prevl[:, t, :] + dprev_h_t,
                                                                                    cache[t])
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

            # 本层得出的dx，作为下一层的prev_l
            dh_prevl = dx

            dweight = (dWx, dWh, db)
            dweights.append(dweight)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回x误差和各层参数误差
        return dx, dweights

    def rnn_step_forward(self, x, prev_h, Wx, Wh, b):
        """
        Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
        activation function.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Inputs:
        - x: Input data for this timestep, of shape (N, D).
        - prev_h: Hidden state from previous timestep, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - cache: Tuple of values needed for the backward pass.
        """

        next_h, cache = None, None
        ##############################################################################
        # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
        # hidden state and any values you need for the backward pass in the next_h   #
        # and cache variables respectively.                                          #
        ##############################################################################
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b

        next_h = np.tanh(z)

        dtanh = 1. - next_h * next_h
        cache = (x, prev_h, Wx, Wh, dtanh)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return next_h, cache

    def rnn_step_backward(self, dnext_h, cache):
        """
        Backward pass for a single timestep of a vanilla RNN.

        Inputs:
        - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        - cache: Cache object from the forward pass

        Returns a tuple of:
        - dx: Gradients of input data, of shape (N, D)
        - dprev_h: Gradients of previous hidden state, of shape (N, H)
        - dWx: Gradients of input-to-hidden weights, of shape (D, H)
        - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        - db: Gradients of bias vector, of shape (H,)
        """
        dx, dprev_h, dWx, dWh, db = None, None, None, None, None
        ##############################################################################
        # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
        #                                                                            #
        # HINT: For the tanh function, you can compute the local derivative in terms #
        # of the output value from tanh.                                             #
        ##############################################################################
        x, prev_h, Wx, Wh, dtanh = cache
        dz = dnext_h * dtanh
        dx = Tools.matmul(dz, Wx.T)
        dprev_h = Tools.matmul(dz, Wh.T)
        dWx = Tools.matmul(x.T, dz)
        dWh = Tools.matmul(prev_h.T, dz)
        db = np.sum(dz, axis=0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return dx, dprev_h, dWx, dWh, db


# LSTM 类
class LSTMLayer(object):
    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.lstmParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        lstmParams = []
        for layer in range(layersNum):
            Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            b = np.zeros(4 * H, dataType)

            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})
        self.lstmParams = lstmParams


    def _initNnWeightOrthogonal(self, D , H, layersNum, dataType):

        # 层次
        lstmParams = []
        for layer in range(layersNum):
            # Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            Wh =  Tools.initOrthogonal( (H, 4*H),self.init_rng, dataType)
            DH = D if 0 == layer else H
            # Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, 4 * H)).astype(dataType)
            Wx =  Tools.initOrthogonal( (DH, 4*H),self.init_rng, dataType)
            b = np.zeros(4 * H, dataType)
            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})

        self.lstmParams = lstmParams

    # 预测时前向传播
    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)
        return self.out

    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            # self._initNnWeight(D, H, L, self.dataType)
            self._initNnWeightOrthogonal(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.lstm_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    # TODO 实现反向传播逻辑，先按照时间，再按照层次，再更新Wx/Wf/b/V/bv 及偏置的反向传播梯度
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        N, T, D = input.shape
        H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.lstm_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'])
            # self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'] = self.optimizerObjs[l].getUpdWeights(w, dw[L-1-l], lrt)
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)
            # self.optimizerObjs[l].getUpdWeights(w, dw[l], lrt)

    def lstm_forward(self, x):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        h, cache = None, None
        #############################################################################
        # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = int(self.lstmParams[0]['b'].shape[0] / 4)  # 取整
        xh = x  # 首层输入是x
        for layer in range(L):
            h = np.zeros((N, T, H))
            h0 = np.zeros((N, H))
            c = np.zeros((N, T, H))
            c0 = np.zeros((N, H))
            cache = []
            for t in range(T):
                h[:, t, :], c[:, t, :], tmp_cache = self.lstm_step_forward(xh[:, t, :], h[:, t - 1, :] if t > 0 else h0,
                                                                           c[:, t - 1, :] if t > 0 else c0,
                                                                           self.lstmParams[layer]['Wx'],
                                                                           self.lstmParams[layer]['Wh'],
                                                                           self.lstmParams[layer]['b'])
                cache.append(tmp_cache)
            xh = h  # 之后以h作为xh作为跨层输入
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            self.lstmParams[layer]['h'] = h
            self.lstmParams[layer]['c'] = c
            self.lstmParams[layer]['cache'] = cache
        return h

    def lstm_backward(self, dh):
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H = dh.shape
        x, _, _, _, _, _, _, _, _, _ = self.lstmParams[0]['cache'][0]
        D = x.shape[1]

        dh_prevl = dh
        # 保存各层dwh,dwx,和db
        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):
            # 得到前向传播保存的cache数组
            cache = self.lstmParams[layer]['cache']

            DH = D if layer == 0 else H
            dx = np.zeros((N, T, DH))
            dWx = np.zeros((DH, 4 * H))

            dWh = np.zeros((H, 4 * H))
            db = np.zeros((4 * H))
            dprev_h = np.zeros((N, H))
            dprev_c = np.zeros((N, H))
            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h, dprev_c, dWx_t, dWh_t, db_t = self.lstm_step_backward(dh_prevl[:, t, :] + dprev_h,
                                                                                            dprev_c,
                                                                                            cache[t])  # 注意此处的叠加
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

            # 本层得出的dx，作为下一层的prev_l
            dh_prevl = dx

            dweight = (dWx, dWh, db)
            dweights.append(dweight)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回x误差和各层参数误差
        return dx, dweights

    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, next_c, cache = None, None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z , of shape(N,4H)
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b

        # of shape(N,H)
        i = Tools.sigmoid(z[:, :H])
        f = Tools.sigmoid(z[:, H:2 * H])
        o = Tools.sigmoid(z[:, 2 * H:3 * H])
        g = np.tanh(z[:, 3 * H:])
        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)

        cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return next_h, next_c, cache

    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache

        dnext_c = dnext_c + o * (1 - np.tanh(next_c) ** 2) * dnext_h  # next_h = o*np.tanh(next_c)
        di = dnext_c * g  # next_c = f*prev_c + i*g
        df = dnext_c * prev_c  # next_c = f*prev_c + i*g
        do = dnext_h * np.tanh(next_c)  # next_h = o*np.tanh(next_c)
        dg = dnext_c * i  # next_h = o*np.tanh(next_c)
        dprev_c = f * dnext_c  # next_c = f*prev_c + i*g
        dz = np.hstack((i * (1 - i) * di, f * (1 - f) * df, o * (1 - o) * do, (1 - g ** 2) * dg))  # 共四部分

        dx = Tools.matmul(dz, Wx.T)
        dprev_h = Tools.matmul(dz, Wh.T)
        dWx = Tools.matmul(x.T, dz)
        dWh = Tools.matmul(prev_h.T, dz)

        db = np.sum(dz, axis=0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dprev_c, dWx, dWh, db


#最后一层concate,输出N*T*2H
class BiLSTMLayer(object):
    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    # 直接输出分类维度
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.lstmParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        lstmParams = []
        for layer in range(layersNum):
            Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            iWh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
                iWx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
                iWx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            b = np.zeros(4 * H, dataType)
            ib = np.zeros(4 * H, dataType)

            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b,
                               'iWx': iWx, 'iWh': iWh, 'ib': ib
                               # , 'U': U, 'V': V, 'bc': bc
                               })
        self.lstmParams = lstmParams

    # 预测时前向传播
    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)
        return self.out

    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.lstm_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        # N, T, D = input.shape
        # H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        # dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.lstm_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'],
                 self.lstmParams[l]['iWx'], self.lstmParams[l]['iWh'], self.lstmParams[l]['ib']
                 )
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def lstm_forward(self, x):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        #############################################################################
        # TODO: Implement the forward pass for an BiLSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = int(self.lstmParams[0]['b'].shape[0] / 4)  # 取整
        xh = x  # 首层输入是x
        ixh = x # 反向
        for layer in range(L):
            h0 = np.zeros((N, H))
            c0 = np.zeros((N, H))

            # 右向
            h = np.zeros((N, T, H))
            c = np.zeros((N, T, H))
            cache = []

            # 左向
            ih = np.zeros((N, T, H))
            ic = np.zeros((N, T, H))
            icache = []
            for t in range(T):
                # 右向
                h[:, t, :], c[:, t, :], tmp_cache = self.lstm_step_forward(xh[:, t, :],
                                                                           h[:, t - 1, :] if t > 0 else h0,
                                                                           c[:, t - 1, :] if t > 0 else c0,
                                                                           self.lstmParams[layer]['Wx'],
                                                                           self.lstmParams[layer]['Wh'],
                                                                           self.lstmParams[layer]['b'])
                cache.append(tmp_cache)

                # 左向,
                # 若此处ih和x的下标保持一致，均由大到小排列，后续无需倒排,提高效率
                ih[:, T - 1 - t, :], ic[:, T - 1 - t, :], tmp_icache = self.lstm_step_forward(ixh[:, T - 1 - t, :],
                                                                              ih[:, T - t, :] if t > 0 else h0,
                                                                              ic[:, T - t, :] if t > 0 else c0,
                                                                              self.lstmParams[layer]['iWx'],
                                                                              self.lstmParams[layer]['iWh'],
                                                                              self.lstmParams[layer]['ib'])

                # icache下标和ih下标是反向对应的                                                              self.lstmParams[layer]['ib'])
                icache.append(tmp_icache)

            # 右向
            self.lstmParams[layer]['h'] = h
            self.lstmParams[layer]['c'] = c
            self.lstmParams[layer]['cache'] = cache

            # 左向
            self.lstmParams[layer]['ih'] = ih
            self.lstmParams[layer]['ic'] = ic
            self.lstmParams[layer]['icache'] = icache

            # Batch * TimeStep * H
            xh = h
            ixh = ih
            self.lstmParams[layer]['xh'] = xh
            self.lstmParams[layer]['ixh'] = ixh

        xh_final = np.concatenate((xh,ixh),axis=2) # 在H维度上做拼接
        self.lstmParams[layer]['xh_final'] = xh_final

        return xh_final

    def lstm_backward(self, dh_all):
        """
        Backward pass for an BiLSTM over an entire sequence of data.]

        Inputs:
        - dh_all: Upstream gradients of hidden states, of shape (N, T, 2*H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        #############################################################################
        # TODO: Implement the backward pass for an BiLSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H_time_2 = dh_all.shape #得到的误差是batch *T* 2H
        H = int(H_time_2 / 2)

        x, _, _, _, _, _, _, _, _, _ = self.lstmParams[0]['cache'][0]

        D = x.shape[1] # 单个时间步上维度
        dh = dh_all[:,:,0:H]
        dih = dh_all[:,:,H:2*H]
        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):

            dh_prevl = dh
            dih_prevl = dih

            DH = D if layer == 0 else H
            # 右向
            dx = np.zeros((N, T, DH))

            cache = self.lstmParams[layer]['cache']
            dWx = np.zeros((DH, 4 * H))
            dWh = np.zeros((H, 4 * H))
            db = np.zeros((4 * H))
            dprev_h = np.zeros((N, H))
            dprev_c = np.zeros((N, H))

            # 左向
            dix = np.zeros((N, T, DH))
            icache = self.lstmParams[layer]['icache']
            diWx = np.zeros((DH, 4 * H))
            diWh = np.zeros((H, 4 * H))
            dib = np.zeros((4 * H))
            dprev_ih = np.zeros((N, H))
            dprev_ic = np.zeros((N, H))

            for t in range(T - 1, -1, -1):
                # 右向
                dx[:, t, :], dprev_h, dprev_c, dWx_t, dWh_t, db_t = self.lstm_step_backward(dh_prevl[:, t, :] + dprev_h,
                                                                                            dprev_c,
                                                                                            cache[t])  # 注意此处的叠加
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

                # fwd选择ih和输入x的下标一致对应，且之后合并前馈时，ih按时间步一致再前馈
                # bp时,按照时间步倒序回传,dih从小到大回传
                dix[:, T - 1 - t, :], dprev_ih, dprev_ic, diWx_t, diWh_t, db_it = self.lstm_step_backward(dih_prevl[:, T - 1 - t, :] + dprev_ih,
                                                                                                  dprev_ic,
                                                                                                  # icache[T - 1 - t])  # 注意此处的叠加
                                                                                                  icache[t])  # 注意此处的叠加
                diWx += diWx_t
                diWh += diWh_t
                dib += db_it

            dweight = (dWx, dWh, db, diWx, diWh, dib)
            dweights.append(dweight)

            # 本层得出的dx，作为下一层的误差输入
            dh = dx
            dih = dix
        # 第一层，正反两个方向的误差相加，得到总的dx返回上一层
        # 如果rnn是第一层，则误差不需要继续向上传递
        # 返回x误差和各层参数误差

        dh_t_all = dh + dih # 合并得到dx
        return dh_t_all, dweights

    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, next_c, cache = None, None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z , of shape(N,4H)
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b

        # of shape(N,H)
        i = Tools.sigmoid(z[:, :H])
        f = Tools.sigmoid(z[:, H:2 * H])
        o = Tools.sigmoid(z[:, 2 * H:3 * H])
        g = np.tanh(z[:, 3 * H:])
        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)

        cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)

        return next_h, next_c, cache

    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache

        dnext_c = dnext_c + o * (1 - np.tanh(next_c) ** 2) * dnext_h  # next_h = o*np.tanh(next_c)
        di = dnext_c * g  # next_c = f*prev_c + i*g
        df = dnext_c * prev_c  # next_c = f*prev_c + i*g
        do = dnext_h * np.tanh(next_c)  # next_h = o*np.tanh(next_c)
        dg = dnext_c * i  # next_h = o*np.tanh(next_c)
        dprev_c = f * dnext_c  # next_c = f*prev_c + i*g
        dz = np.hstack((i * (1 - i) * di, f * (1 - f) * df, o * (1 - o) * do, (1 - g ** 2) * dg))  # 共四部分

        dx = Tools.matmul(dz, Wx.T)
        dprev_h = Tools.matmul(dz, Wh.T)
        dWx = Tools.matmul(x.T, dz)
        dWh = Tools.matmul(prev_h.T, dz)

        db = np.sum(dz, axis=0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dprev_c, dWx, dWh, db

#最后一层concate,输出N*T*2H
class BiLSTMLayer_succ(object):
    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    # 直接输出分类维度
    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.lstmParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        lstmParams = []
        for layer in range(layersNum):
            Wh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            iWh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
                iWx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 4 * H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
                iWx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 4 * H)).astype(dataType)
            b = np.zeros(4 * H, dataType)
            ib = np.zeros(4 * H, dataType)

            # 合并两个方向输出的前馈参数
            # U = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            # V = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            # bc = np.zeros(H, dataType)
            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b,
                               'iWx': iWx, 'iWh': iWh, 'ib': ib
                               # , 'U': U, 'V': V, 'bc': bc
                               })
        self.lstmParams = lstmParams

    # 预测时前向传播
    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)
        return self.out

    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.lstm_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        N, T, D = input.shape
        H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        # dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.lstm_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'],
                 self.lstmParams[l]['iWx'], self.lstmParams[l]['iWh'], self.lstmParams[l]['ib']
                 # ,self.lstmParams[l]['U'], self.lstmParams[l]['V'], self.lstmParams[l]['bc']
                 # self.lstmParams[l]['U'],  self.lstmParams[l]['bc']
                 )
            # self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'] = self.optimizerObjs[l].getUpdWeights(w, dw[L-1-l], lrt)
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def lstm_forward(self, x):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        # 右向
        h, cache = None, None

        # 左向
        ih, icache = None, None

        #############################################################################
        # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = int(self.lstmParams[0]['b'].shape[0] / 4)  # 取整
        xh = x  # 首层输入是x
        ixh = x # 反向
        for layer in range(L):
            h0 = np.zeros((N, H))
            c0 = np.zeros((N, H))

            # 右向
            h = np.zeros((N, T, H))
            c = np.zeros((N, T, H))
            cache = []

            # 左向
            ih = np.zeros((N, T, H))
            ic = np.zeros((N, T, H))
            icache = []
            for t in range(T):
                # 右向
                h[:, t, :], c[:, t, :], tmp_cache = self.lstm_step_forward(xh[:, t, :],
                                                                           h[:, t - 1, :] if t > 0 else h0,
                                                                           c[:, t - 1, :] if t > 0 else c0,
                                                                           self.lstmParams[layer]['Wx'],
                                                                           self.lstmParams[layer]['Wh'],
                                                                           self.lstmParams[layer]['b'])
                cache.append(tmp_cache)

                # 左向,
                # 若此处ih和x的下标保持一致，均由大到小排列，后续无需倒排，但此处逻辑不易读
                ih[:, T - 1 - t, :], ic[:, T - 1 - t, :], tmp_icache = self.lstm_step_forward(ixh[:, T - 1 - t, :],
                                                                              ih[:, T - t, :] if t > 0 else h0,
                                                                              ic[:, T - t, :] if t > 0 else c0,
                                                                              self.lstmParams[layer]['iWx'],
                                                                              self.lstmParams[layer]['iWh'],
                                                                              self.lstmParams[layer]['ib'])

                # 若ih和输入x的下标逆序对应，后面合并前馈时，需要ih按时间步倒排，选这个方法
                # ih[:, t, :], ic[:, t, :], tmp_icache = self.lstm_step_forward(xh[:, T - t - 1, :],
                #                                                               ih[:, t - 1, :] if t > 0 else h0,
                #                                                               ic[:, t - 1, :] if t > 0 else c0,
                #                                                               self.lstmParams[layer]['iWx'],
                #                                                               self.lstmParams[layer]['iWh'],
                # icache下标和ih下标是反向对应的                                                              self.lstmParams[layer]['ib'])

                icache.append(tmp_icache)

            # 右向
            self.lstmParams[layer]['h'] = h
            self.lstmParams[layer]['c'] = c
            self.lstmParams[layer]['cache'] = cache

            # 左向
            self.lstmParams[layer]['ih'] = ih
            self.lstmParams[layer]['ic'] = ic
            self.lstmParams[layer]['icache'] = icache

            # 合并
            # U = self.lstmParams[layer]['U']
            # V = self.lstmParams[layer]['V']
            # bc = self.lstmParams[layer]['bc']

            # 左向输出,ih和输入x的下标逆序对应，合并前馈时，需要ih在时间步维度上倒排，
            # ih_t = ih[:, ::-1, :]
            # 左向输出,ih和x下标保持一致，均由大到小排列，后续无需倒排
            # ih_t = ih
            # 使用ReLU激活
            # xh =ReLU.activate( Tools.matmul(h, U) + Tools.matmul(ih, V) + bc)  # 之后以h作为xh作为跨层输入
            # xh =Tools.sigmoid( Tools.matmul(h, U) + Tools.matmul(ih, V) + bc)  # 之后以h作为xh作为跨层输入
            # xh =np.tanh( Tools.matmul(h, U) + Tools.matmul(ih, V) + bc)  # 之后以h作为xh作为跨层输入
            # hih = np.concatenate((h,ih),2)
            # xh = Tools.matmul(hih,U) + bc
            # Batch * TimeStep * H
            xh = h
            ixh = ih
            self.lstmParams[layer]['xh'] = xh
            self.lstmParams[layer]['ixh'] = ixh

            # if layer == L -1: # 最后一层，做concat合并,且只保留最后一个时间步的数据
            #     # U = self.lstmParams[layer]['U']
            #     # V = self.lstmParams[layer]['V']
            #     # bc = self.lstmParams[layer]['bc']
            #
            #     # xh_ori = Tools.matmul(h, U) + Tools.matmul(ih, V) + bc
            #     # xh_final =ReLU.activate( xh_ori )
            #     # 只取最右边一个时间步，再合并h维度，得到batch * 2H
            #     # xh_final = np.concatenate((xh[:,-1,:],ixh[:,0,:]),axis=1)
            #     # xh_final = np.concatenate((xh[:,-1,:],ixh[:,0,:]),axis=1)
            #     # r_ixh = ixh[:,::-1,:] # 按照时间维度逆序
            #     # xh_final = np.concatenate((xh,r_ixh),axis=2) # 在H维度上做拼接
            #     xh_final = np.concatenate((xh,ixh),axis=2) # 在H维度上做拼接
            #
            #     self.lstmParams[layer]['xh_final'] = xh_final
            #     # self.lstmParams[layer]['T'] = T
            #     # self.lstmParams[layer]['xh_ori'] = xh_ori
        xh_final = np.concatenate((xh,ixh),axis=2) # 在H维度上做拼接
        self.lstmParams[layer]['xh_final'] = xh_final

        return xh_final
        # return ih

    def lstm_backward(self, dh_all):
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dh_all: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        # 右向
        # dx, dh0, dWx, dWh, db = None, None, None, None, None
        # 左向
        # dix, dih0, diWx, diWh, dib = None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H_time_2 = dh_all.shape #得到的误差是batch *T* 2H
        H = int(H_time_2 / 2)
        # N, T, H = dh_all.shape # 逆向
        # dh = dh_all
        # dih = dh_all

        x, _, _, _, _, _, _, _, _, _ = self.lstmParams[0]['cache'][0]

        D = x.shape[1] # 单个时间步上维度
        # T = self.lstmParams[self.layersNum - 1]['T']
        # U = self.lstmParams[self.layersNum - 1]['U']
        # V = self.lstmParams[self.layersNum - 1]['V']
        # xh_final =self.lstmParams[self.layersNum - 1]['xh_final']
        # xh_ori =self.lstmParams[self.layersNum - 1]['xh_ori']
        # dhih = dh_all*Tools.bp4tanh(xh_final)
        # dhih = ReLU.bp(dh_all,xh_ori)

        # dh = np.zeros((N,T,H),dtype=self.dataType)
        # dih = np.zeros((N, T, H), dtype=self.dataType)
        # dh[:,:,:] = dh_all[:,:,0:H]
        # r_dih[:,:,:] = dh_all[:,:,H:2*H]
        dh = dh_all[:,:,0:H]
        dih = dh_all[:,:,H:2*H]
        # # r_dih = dh_all[:,:,H:2*H]
        # dih = r_dih[:,:,::-1] # 在H上倒序，得到原始dih

        # dh = Tools.matmul(dhih,U.T)
        # dih = Tools.matmul(dhih, V.T)

        #
        # # dU = Tools.matmul(,dh_all)
        # # dV = Tools.matmul(,dh_all)
        # # dbc = np.sum(dh_all, axis=0)
        #
        # dh_prevl = dh
        # dih_prevl = dih
        # 保存各层dwh,dwx,和db
        # dh_t_all = dh_all
        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):

            # U = self.lstmParams[layer]['U']
            # V = self.lstmParams[layer]['V']
            h = self.lstmParams[layer]['h']
            ih = self.lstmParams[layer]['ih']
            # xh = self.lstmParams[layer]['xh']

            # 激活函数bp
            # dh_t_all = ReLU.bp(dh_t_all,xh)
            # dh_t_all = dh_t_all * Tools.bp4sigmoid(xh)
            # dh_t_all = dh_t_all * Tools.bp4tanh(xh)
            # dh = Tools.matmul(dh_t_all, U.T)
            # dih = Tools.matmul(dh_t_all, V.T)
            # dhih = Tools.matmul(dh_t_all, U.T)  #(N,T,2H)
            # dh = dhih[:,:,0:H]
            # dih = dhih[:,:,H:2*H]

            # dh = dh_t_all
            # dih = dh_t_all

            dh_prevl = dh
            dih_prevl = dih

            DH = D if layer == 0 else H

            # 右向
            dx = np.zeros((N, T, DH))

            cache = self.lstmParams[layer]['cache']
            dWx = np.zeros((DH, 4 * H))
            dWh = np.zeros((H, 4 * H))
            db = np.zeros((4 * H))
            dprev_h = np.zeros((N, H))
            dprev_c = np.zeros((N, H))

            # 左向
            dix = np.zeros((N, T, DH))
            icache = self.lstmParams[layer]['icache']
            diWx = np.zeros((DH, 4 * H))
            diWh = np.zeros((H, 4 * H))
            dib = np.zeros((4 * H))
            dprev_ih = np.zeros((N, H))
            dprev_ic = np.zeros((N, H))

            # 前馈合并参数
            # xht = (self.lstmParams[layer]['xh'])
            # dU = np.zeros((H, H))
            # dV = np.zeros((H, H))
            # dbc_final = np.zeros((H))

            for t in range(T - 1, -1, -1):
                # 右向
                dx[:, t, :], dprev_h, dprev_c, dWx_t, dWh_t, db_t = self.lstm_step_backward(dh_prevl[:, t, :] + dprev_h,
                                                                                            dprev_c,
                                                                                            cache[t])  # 注意此处的叠加
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

                # 左向
                # # fwd选择ih和输入x的下标逆序对应，且之后合并前馈时，ih按时间步倒排为ih_t再前馈
                # # bp时,按照时间步倒序回传,dih变成从小到大回传
                # dix[:, t, :], dprev_ih, dprev_ic, diWx_t, diWh_t, db_it = self.lstm_step_backward(dih_prevl[:, T - 1 - t, :] + dprev_ih,
                #                                                                                   dprev_ic,

                # fwd选择ih和输入x的下标一致对应，且之后合并前馈时，ih按时间步一致再前馈
                # bp时,按照时间步倒序回传,dih从小到大回传
                dix[:, T - 1 - t, :], dprev_ih, dprev_ic, diWx_t, diWh_t, db_it = self.lstm_step_backward(dih_prevl[:, T - 1 - t, :] + dprev_ih,
                                                                                                  dprev_ic,
                                                                                                  # icache[T - 1 - t])  # 注意此处的叠加
                                                                                                  icache[t])  # 注意此处的叠加
                diWx += diWx_t
                diWh += diWh_t
                dib += db_it

                # if layer == L-1: # 最上一层,计算dU和dV
                # # 层前馈合并参数
                #     dU += Tools.matmul(h[:, t, :].T, dh[:, t, :])
                #     dV += Tools.matmul(ih[:, t, :].T, dih[:, t, :])
                    # dbc += dh_all[:, t, :]

                # dU += Tools.matmul(h[:, t, :].T, dh_all[:, t, :])
                # dV += Tools.matmul(ih[:, t, :].T, dh_all[:, t, :])
                # dbc += dh_all[:, t, :]

            # dU += np.concatenate((Tools.matmul((h[:, t, :]).T, dh[:, t, :]),
            #                       (Tools.matmul((ih[:, t, :]).T, dih[:, t, :]))),0)
            # pass

            # 需不需要给梯度除以批量,不需要，误差已经除过了
            # dU = dU / N
            # dV = dV / N
            # if layer == T - 1:  # 最上一层,计算dU和dV
            #     dbc_final = np.sum(np.sum(dh_all, axis=0), axis=0)

            # dweight = (dWx, dWh, db, diWx, diWh, dib, dU, dV, dbc_final)
            # dweight = (dWx, dWh, db, diWx, diWh, dib, dU, dbc_final)
            dweight = (dWx, dWh, db, diWx, diWh, dib)
            dweights.append(dweight)

            # 本层得出的dx，作为下一层的误差输入
            # dh_t_all = dx + dix
            dh = dx
            dih = dix
        # 第一层，正反两个方向的误差相加，得到总的dx返回上一层
        # 如果rnn是第一层，则误差不需要继续向上传递了
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回x误差和各层参数误差
        # r_dih = dih[:,:,::-1] # dhi逆序
        # dh_t_all = dh + r_dih # 合并得到dx
        dh_t_all = dh + dih # 合并得到dx
        return dh_t_all, dweights
        # return dih, dweights

    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, next_c, cache = None, None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z , of shape(N,4H)
        z = Tools.matmul(x, Wx) + Tools.matmul(prev_h, Wh) + b

        # of shape(N,H)
        i = Tools.sigmoid(z[:, :H])
        f = Tools.sigmoid(z[:, H:2 * H])
        o = Tools.sigmoid(z[:, 2 * H:3 * H])
        g = np.tanh(z[:, 3 * H:])
        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)

        cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return next_h, next_c, cache

    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache

        dnext_c = dnext_c + o * (1 - np.tanh(next_c) ** 2) * dnext_h  # next_h = o*np.tanh(next_c)
        di = dnext_c * g  # next_c = f*prev_c + i*g
        df = dnext_c * prev_c  # next_c = f*prev_c + i*g
        do = dnext_h * np.tanh(next_c)  # next_h = o*np.tanh(next_c)
        dg = dnext_c * i  # next_h = o*np.tanh(next_c)
        dprev_c = f * dnext_c  # next_c = f*prev_c + i*g
        dz = np.hstack((i * (1 - i) * di, f * (1 - f) * df, o * (1 - o) * do, (1 - g ** 2) * dg))  # 共四部分

        dx = Tools.matmul(dz, Wx.T)
        dprev_h = Tools.matmul(dz, Wh.T)
        dWx = Tools.matmul(x.T, dz)
        dWh = Tools.matmul(prev_h.T, dz)

        db = np.sum(dz, axis=0)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dprev_c, dWx, dWh, db


# GRU 类
class GRULayer(object):

    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        # self.optimizer = optimizer
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.gruParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        gruParams = []
        for layer in range(layersNum):
            Wzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            War = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            if (0 == layer):
                Wzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, 2 * H)).astype(dataType)
                Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, H)).astype(dataType)
            else:
                Wzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
                Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            bz = np.zeros(2 * H, dataType)
            ba = np.zeros(H, dataType)
            gruParams.append({'Wzx': Wzx, 'Wzh': Wzh, 'bz': bz, 'Wax': Wax, 'War': War, 'ba': ba})

        self.gruParams = gruParams

    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)

        return self.out

    # 预测时前向传播
    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)
        h = self.gru_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        # self.out = h[:,-1,:]
        self.out = h
        return self.out

    # 反向传播方法(误差和权参)
    # TODO 实现反向传播逻辑，先按照时间，再按照层次，再更新Wx/Wf/b/V/bv 及偏置的反向传播梯度
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        N, T, D = input.shape
        H = delta.shape[1]
        # 只有最后一个T填delta，其余的dh梯度设置为0
        dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        dh = delta
        dx, dweight = self.gru_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'], self.gruParams[l]['Wax'],
                 self.gruParams[l]['War'], self.gruParams[l]['ba'])
            # self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'] = self.optimizerObjs[l].getUpdWeights(w, dw[L-1-l], lrt)
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def gru_forward(self, x):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        h, cache = None, None
        #############################################################################
        # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = self.gruParams[0]['ba'].shape[0]  # 取整
        xh = x  # 首层输入是x
        for layer in range(L):
            h = np.zeros((N, T, H))
            h0 = np.zeros((N, H))
            cache = []
            for t in range(T):
                h[:, t, :], tmp_cache = self.gru_step_forward(xh[:, t, :], h[:, t - 1, :] if t > 0 else h0,
                                                              self.gruParams[layer]['Wzx'],
                                                              self.gruParams[layer]['Wzh'],
                                                              self.gruParams[layer]['bz'],
                                                              self.gruParams[layer]['Wax'],
                                                              self.gruParams[layer]['War'],
                                                              self.gruParams[layer]['ba'],
                                                              )
                cache.append(tmp_cache)
            xh = h  # 之后以h作为xh作为跨层输入
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            self.gruParams[layer]['h'] = h
            self.gruParams[layer]['cache'] = cache

        return h

    def gru_backward(self, dh):
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H = dh.shape
        x, _, _, _, _, _, _, _, _, _ = self.gruParams[0]['cache'][0]
        D = x.shape[1]

        dh_prevl = dh
        # 保存各层dwh,dwx,和db
        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):
            # 得到前向传播保存的cache数组
            cache = self.gruParams[layer]['cache']

            DH = D if layer == 0 else H
            dx = np.zeros((N, T, DH))
            dWzx = np.zeros((DH, 2 * H))
            dWzh = np.zeros((H, 2 * H))
            dbz = np.zeros((2 * H))

            dWax = np.zeros((DH, H))
            dWar = np.zeros((H, H))
            dba = np.zeros((H))

            dprev_h = np.zeros((N, H))

            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h, dWzx_t, dWzh_t, dbz_t, dWax_t, dWar_t, dba_t = self.gru_step_backward(
                    dh_prevl[:, t, :] + dprev_h,
                    cache[t])  # 注意此处的叠加
                dWzx += dWzx_t
                dWzh += dWzh_t
                dbz += dbz_t

                dWax += dWax_t
                dWar += dWar_t
                dba += dba_t
            # 本层得出的dx，作为下一层的prev_l
            dh_prevl = dx

            dweight = (dWzx, dWzh, dbz, dWax, dWar, dba)
            dweights.append(dweight)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回x误差和各层参数误差
        return dx, dweights

    def gru_step_forward(self, x, prev_h, Wzx, Wzh, bz, Wax, War, ba):
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wzx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, cache = None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z_hat, of shape(N,4H)
        z_hat = Tools.matmul(x, Wzx) + Tools.matmul(prev_h, Wzh) + bz

        # of shape(N,H)
        r = Tools.sigmoid(z_hat[:, :H])
        z = Tools.sigmoid(z_hat[:, H:2 * H])

        a = Tools.matmul(x, Wax) + Tools.matmul(r * prev_h, War) + ba

        next_h = prev_h * (1. - z) + z * np.tanh(a)

        cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return next_h, cache

    def gru_step_backward(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh*(tanha-prev_h)
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))

        dhat_a = Tools.matmul(da, War.T)
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)

        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)

        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)

        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_succ(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        # dz_hat_2 = dz*(z*(1.-z))
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))
        # dz_hat_1 = prev_h * Tools.matmul(dh*z*(1-tanha**2), War.T)*(r*(1.-r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)
        dx_23 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34  #dh_prev_3 + dh_prev_4
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_23 # +dx_3
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v2(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        # dz_hat_2 = dz*(z*(1.-z))
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))
        # dz_hat_1 = prev_h * Tools.matmul(dh*z*(1-tanha**2), War.T)*(r*(1.-r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dx_23 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_34  # dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_23  # +dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v1(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha * tanha)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        dz_hat_2 = dz * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dx_2 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_34  # dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2  # +dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v0(self, dnext_h, cache):
        """
        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha * tanha)
        dh_prev_1 = dh * (1. - z)
        dz = dh * (z + tanha)
        dz_hat_2 = dz * (z * (1. - z))

        d13 = np.matmul(da, War.T)
        dr = d13 * prev_h
        dx_1 = np.matmul(da, Wax.T)
        dh_prev_2 = d13 * r
        dz_hat_1 = dh_prev_2 * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dh_prev_3 = np.matmul(dz_hat, Wzh.T)
        dx_2 = np.matmul(dz_hat, Wzx.T)
        dx_3 = np.matmul(dz_hat, Wzx.T)
        dh_prev_4 = np.matmul(dz_hat, Wzh.T)
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = np.matmul(x.T, da)
        dWar = np.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = np.matmul(x.T, dz_hat)
        dWzh = np.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

# concat 方式
class BiGRULayer(object):

    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.gruParams = []

        # 不需要保存各层中间产出的 st和f(st)，已经在前反向传播中保留，用于前向和反向传播，
        self.deltaPrev = []  # 上一层激活后的误差输出

    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        gruParams = []
        for layer in range(layersNum):
            # 右向
            Wzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            War = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            # 左向
            iWzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            iWar = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)

            DH = D if layer == 0 else H
            # 右向
            Wzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, 2 * H)).astype(dataType)
            Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, H)).astype(dataType)
            bz = np.zeros(2 * H, dataType)
            ba = np.zeros(H, dataType)

            # 左向
            iWzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, 2 * H)).astype(dataType)
            iWax = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, H)).astype(dataType)
            ibz = np.zeros(2 * H, dataType)
            iba = np.zeros(H, dataType)

            gruParams.append({'Wzx': Wzx, 'Wzh': Wzh, 'bz': bz, 'Wax': Wax, 'War': War, 'ba': ba,
                              'iWzx': iWzx, 'iWzh': iWzh, 'ibz': ibz, 'iWax': iWax, 'iWar': iWar, 'iba': iba
                              })
        self.gruParams = gruParams

    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)

        return self.out

    # 预测时前向传播
    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)

        # N进 v 1出 模型，只保留时序最后的一项输出
        # h = self.gru_forward(x)
        # self.out = h[:,-1,:]

        # N进N出模型，全部输出
        self.out = self.gru_forward(x)
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        # N v 1 只有最后一个T填delta，其余的dh梯度设置为0
        # N, T, D = input.shape
        # H = delta.shape[1]
        # dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta

        # N v N模型
        dh = delta
        dx, dweight = self.gru_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            # w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],
            #      self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'],
            #      self.gruParams[l]['iWzx'], self.gruParams[l]['iWzh'], self.gruParams[l]['ibz'],
            #      self.gruParams[l]['iWax'], self.gruParams[l]['iWar'], self.gruParams[l]['iba']
            #      # ,self.gruParams[l]['U'], self.gruParams[l]['V'], self.gruParams[l]['bc']
            #      )
            params=self.gruParams[l]
            w = (params['Wzx'], params['Wzh'], params['bz'],
                 params['Wax'], params['War'], params['ba'],
                 params['iWzx'], params['iWzh'], params['ibz'],
                 params['iWax'], params['iWar'], params['iba']
                 )

            # is_same0 = op.eq(w,w0)

            # 梯度倒序append到dw中
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def bpWeights_v1(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],
                 self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'],
                 self.gruParams[l]['iWzx'], self.gruParams[l]['iWzh'], self.gruParams[l]['ibz'],
                 self.gruParams[l]['iWax'], self.gruParams[l]['iWar'], self.gruParams[l]['iba']
                 # ,self.gruParams[l]['U'], self.gruParams[l]['V'], self.gruParams[l]['bc']
                 )
            w1 = (self.gruParams[l][element] for element in self.gruParams[l])
            w2 = (v for (k,v) in self.gruParams[l].items())
            is_same1 = op.eq(w,w1)
            is_same2 = op.eq(w, w2)

            # 梯度倒序append到dw中
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def gru_forward(self, x):
        """
        Forward pass for an BiGRU over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """

        #############################################################################
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = self.gruParams[0]['ba'].shape[0]  # 取整
        xh = x  # 首层输入是x
        ixh= x  # 反向
        for layer in range(L):
            h0 = np.zeros((N, H))

            # 右向
            h = np.zeros((N, T, H))
            cache = []

            # 左向
            ih = np.zeros((N, T, H))
            icache = []
            for t in range(T):
                # 右向
                h[:, t, :], tmp_cache = self.gru_step_forward(xh[:, t, :],
                                                              h[:, t - 1, :] if t > 0 else h0,
                                                              self.gruParams[layer]['Wzx'],
                                                              self.gruParams[layer]['Wzh'],
                                                              self.gruParams[layer]['bz'],
                                                              self.gruParams[layer]['Wax'],
                                                              self.gruParams[layer]['War'],
                                                              self.gruParams[layer]['ba']
                                                              )
                cache.append(tmp_cache)

                # 左向
                ih[:, T - 1 - t , :], tmp_icache = self.gru_step_forward(ixh[:, T - 1 - t, :],
                                                              ih[:, T - t, :] if t > 0 else h0,
                                                              self.gruParams[layer]['iWzx'],
                                                              self.gruParams[layer]['iWzh'],
                                                              self.gruParams[layer]['ibz'],
                                                              self.gruParams[layer]['iWax'],
                                                              self.gruParams[layer]['iWar'],
                                                              self.gruParams[layer]['iba']
                                                              )
                # icache是下标和ih下标是返向的                                              )
                icache.append(tmp_icache)
            # 右向
            self.gruParams[layer]['h'] = h
            self.gruParams[layer]['cache'] = cache

            # 左向
            self.gruParams[layer]['ih'] = ih
            self.gruParams[layer]['icache'] = icache

            xh = h
            ixh = ih
        xh_final = np.concatenate((xh, ixh), axis=2)  # 在H维度上做拼接
        # self.gruParams[layer]['xh_final'] = xh_final
        return xh_final

    def gru_backward(self, dh_all):
        """
        Backward pass for an BiLSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, 2H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        diWzx, diWzh, dibz, diWax, diWar, diba = None, None, None, None, None, None
        dU, dV, dbc = None, None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H_time_2 = dh_all.shape #得到的误差是batch *T* 2H
        H = int(H_time_2 / 2)
        # N, T, H = dh_all.shape
        x, _, _, _, _, _, _, _, _, _ = self.gruParams[0]['cache'][0]
        D = x.shape[1]

        dh = dh_all[:,:,0:H]
        dih = dh_all[:,:,H:2*H]

        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):

            dh_prevl = dh
            dih_prevl = dih

            DH = D if layer == 0 else H

            # 右向 得到前向传播保存的cache数组
            cache = self.gruParams[layer]['cache']

            dx = np.zeros((N, T, DH))
            dWzx = np.zeros((DH, 2 * H))
            dWzh = np.zeros((H, 2 * H))
            dbz = np.zeros((2 * H))

            dWax = np.zeros((DH, H))
            dWar = np.zeros((H, H))
            dba = np.zeros((H))

            dprev_h = np.zeros((N, H))

            # 左向
            icache = self.gruParams[layer]['icache']

            dix = np.zeros((N, T, DH))
            diWzx = np.zeros((DH, 2 * H))
            diWzh = np.zeros((H, 2 * H))
            dibz = np.zeros((2 * H))

            diWax = np.zeros((DH, H))
            diWar = np.zeros((H, H))
            diba = np.zeros((H))

            dprev_ih = np.zeros((N, H))

            for t in range(T - 1, -1, -1):
                # 右向
                dx[:, t, :], dprev_h, dWzx_t, dWzh_t, dbz_t, dWax_t, dWar_t, dba_t = self.gru_step_backward(
                    dh_prevl[:, t, :] + dprev_h,
                    cache[t])  # 注意此处的叠加
                dWzx += dWzx_t
                dWzh += dWzh_t
                dbz += dbz_t

                dWax += dWax_t
                dWar += dWar_t
                dba += dba_t

                dix[:, T-1-t, :], dprev_ih, diWzx_t, diWzh_t, dibz_t, diWax_t, diWar_t, diba_t = self.gru_step_backward(
                    dih_prevl[:, T- 1 - t, :] + dprev_ih,
                    icache[t])  # 注意此处的叠加，逆序

                diWzx += diWzx_t
                diWzh += diWzh_t
                dibz += dibz_t

                diWax += diWax_t
                diWar += diWar_t
                diba += diba_t

            dweight = (dWzx, dWzh, dbz, dWax, dWar, dba,
                       diWzx, diWzh, dibz, diWax, diWar, diba
                       #,dU, dV, dbc_final
                       )
            dweights.append(dweight)

            dh = dx
            dih = dix

        dh_t_all = dh + dih # 合并得到dx
        return dh_t_all, dweights

    def gru_step_forward(self, x, prev_h, Wzx, Wzh, bz, Wax, War, ba):
        """
        Forward pass for a single timestep of an GRU&BiGRU.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wzx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        # next_h, cache = None, None
        #############################################################################
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z_hat, of shape(N,4H)
        z_hat = Tools.matmul(x, Wzx) + Tools.matmul(prev_h, Wzh) + bz

        # of shape(N,H)
        r = Tools.sigmoid(z_hat[:, :H])
        z = Tools.sigmoid(z_hat[:, H:2 * H])

        a = Tools.matmul(x, Wax) + Tools.matmul(r * prev_h, War) + ba
        tanha = np.tanh(a)
        next_h = prev_h * (1. - z) + z * tanha

        # cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a, tanha)
        cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, tanha)

        return next_h, cache

    def gru_step_backward(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an GRU&BiGRU.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        # x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a, tanha = cache
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, tanha = cache

        # N, D = x.shape
        H = dnext_h.shape[1]

        # delta
        # tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz_hat_2_n = dh * (tanha - prev_h) * (z * (1. - z))
        dz_hat_2 = (tanha - prev_h) * z * dh_prev_1
        # isSame1 = np.allclose(dz_hat_2,dz_hat_2_n)

        dhat_a = Tools.matmul(da, War.T)
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)

        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)

        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)

        dbz = np.sum(dz_hat, axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_succ(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an GRU&BiGRU.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh*(tanha-prev_h)
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))

        dhat_a = Tools.matmul(da, War.T)
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)

        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)

        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)

        dbz = np.sum(dz_hat, axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

# concat
class BiGRULayer_succ(object):

    def __init__(self, LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dropoutRRate, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        # self.optimizer = optimizer
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False  # 初始化标志

        # dropout 的保留率
        self.dropoutRRate = dropoutRRate
        self.dropoutMask = []

        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        # 初始化w,u,b 和对应偏置,维度，层次和节点个数传参进去。但是没有T，所以不能创建参数
        # 返回的是一个组合结构,按层次（数组）划分的U、W，字典
        # 改为放在首batch X传入时lazy init
        self.gruParams = []

        # 保存各层中间产出的 st和f(st)，用于前向和反向传播
        # 不需要，已经在前反向传播中保留
        self.deltaPrev = []  # 上一层激活后的误差输出

    # N,H,L和优化器在初始化时定义
    # T作为X的一个维度传进来
    # tanh和sigmoid的前反向传播在类内部定义。
    def _initNnWeight(self, D, H, layersNum, dataType):

        # 层次
        gruParams = []
        for layer in range(layersNum):
            # 右向
            Wzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            War = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            # 左向
            iWzh = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, 2 * H)).astype(dataType)
            iWar = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)

            DH = D if layer == 0 else H
            # 右向
            Wzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, 2 * H)).astype(dataType)
            Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, H)).astype(dataType)
            bz = np.zeros(2 * H, dataType)
            ba = np.zeros(H, dataType)

            # 左向
            iWzx = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, 2 * H)).astype(dataType)
            iWax = np.random.uniform(-1 * self.init_rng, self.init_rng, (DH, H)).astype(dataType)
            ibz = np.zeros(2 * H, dataType)
            iba = np.zeros(H, dataType)

            # # 合并前馈
            # U = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            # V = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            # bc = np.zeros(H, dataType)

            gruParams.append({'Wzx': Wzx, 'Wzh': Wzh, 'bz': bz, 'Wax': Wax, 'War': War, 'ba': ba,
                              'iWzx': iWzx, 'iWzh': iWzh, 'ibz': ibz, 'iWax': iWax, 'iWar': iWar, 'iba': iba
                              # ,'U':U, 'V':V, 'bc':bc
                              })

        self.gruParams = gruParams

    def fp(self, input):
        out_tmp = self.inference(input)
        self.out, self.dropoutMask = Tools.dropout4rnn(out_tmp, self.dropoutRRate)

        return self.out

    # 预测时前向传播
    def inference(self, x):
        N, T, D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self._initNnWeight(D, H, L, self.dataType)
            self.isInited = True

        # 缓存已经存入rnnParams里了,此处只需要返回输出结果(N,T,H)

        # N进 v 1出 模型，只保留时序最后的一项输出
        # h = self.gru_forward(x)
        # self.out = h[:,-1,:]

        # N进N出模型，全部输出
        self.out = self.gru_forward(x)
        return self.out

    # 反向传播方法(误差和权参)
    # TODO 实现反向传播逻辑，先按照时间，再按照层次，再更新Wz/Wa/b 的反向传播梯度
    def bp(self, input, delta_ori, lrt):

        if self.dropoutRRate == 1:
            delta = delta_ori
        else:
            delta = delta_ori * self.dropoutMask

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        # N v 1 只有最后一个T填delta，其余的dh梯度设置为0
        # N, T, D = input.shape
        # H = delta.shape[1]
        # dh = np.zeros((N, T, H), self.dataType)
        # dh[:,-1,:] = delta
        # N v N模型
        dh = delta
        dx, dweight = self.gru_backward(dh)

        # 根据梯度更新参数
        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],
                 self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'],
                 self.gruParams[l]['iWzx'], self.gruParams[l]['iWzh'], self.gruParams[l]['ibz'],
                 self.gruParams[l]['iWax'], self.gruParams[l]['iWar'], self.gruParams[l]['iba']
                 # ,self.gruParams[l]['U'], self.gruParams[l]['V'], self.gruParams[l]['bc']
                 )
            # self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'] = self.optimizerObjs[l].getUpdWeights(w, dw[L-1-l], lrt)
            # 梯度倒序append到dw中
            self.optimizerObjs[l].getUpdWeights(w, dw[L - 1 - l], lrt)

    def gru_forward(self, x):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        h, cache = None, None
        ih, icache = None, None
        #############################################################################
        # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        N, T, D = x.shape
        L = self.layersNum
        H = self.gruParams[0]['ba'].shape[0]  # 取整
        xh = x  # 首层输入是x
        ixh= x  # 反向
        # ixh= x[:,::-1,:]  # 反向
        for layer in range(L):
            h0 = np.zeros((N, H))

            # 右向
            h = np.zeros((N, T, H))
            cache = []

            # 左向
            ih = np.zeros((N, T, H))
            icache = []
            for t in range(T):
                # 右向
                h[:, t, :], tmp_cache = self.gru_step_forward(xh[:, t, :],
                                                              h[:, t - 1, :] if t > 0 else h0,
                                                              self.gruParams[layer]['Wzx'],
                                                              self.gruParams[layer]['Wzh'],
                                                              self.gruParams[layer]['bz'],
                                                              self.gruParams[layer]['Wax'],
                                                              self.gruParams[layer]['War'],
                                                              self.gruParams[layer]['ba']
                                                              )
                cache.append(tmp_cache)

                # 左向
                ih[:, T - 1 - t , :], tmp_icache = self.gru_step_forward(ixh[:, T - 1 - t, :],
                                                              ih[:, T - t, :] if t > 0 else h0,
                                                              self.gruParams[layer]['iWzx'],
                                                              self.gruParams[layer]['iWzh'],
                                                              self.gruParams[layer]['ibz'],
                                                              self.gruParams[layer]['iWax'],
                                                              self.gruParams[layer]['iWar'],
                                                              self.gruParams[layer]['iba']
                                                              )
                # ih[:, t, :], tmp_icache = self.gru_step_forward(ixh[:, t, :],
                #                                               ih[:, t - 1, :] if t > 0 else h0,
                #                                               self.gruParams[layer]['iWzx'],
                #                                               self.gruParams[layer]['iWzh'],
                #                                               self.gruParams[layer]['ibz'],
                #                                               self.gruParams[layer]['iWax'],
                #                                               self.gruParams[layer]['iWar'],
                #                                               self.gruParams[layer]['iba']
                # icache是下标和ih下标是返向的                                              )
                icache.append(tmp_icache)
            # 右向
            # xh = h  # 之后以h作为xh作为跨层输入
            self.gruParams[layer]['h'] = h
            self.gruParams[layer]['cache'] = cache

            # 左向
            # ixh = ih  # 之后以h作为xh作为跨层输入
            self.gruParams[layer]['ih'] = ih
            self.gruParams[layer]['icache'] = icache

            # # 合并前馈
            # U = self.gruParams[layer]['U']
            # V = self.gruParams[layer]['V']
            # bc = self.gruParams[layer]['bc']
            #
            # # 左向输出,ih和x下标保持一致，均由大到小排列，后续无需倒排
            # xh = Tools.matmul(h, U) + Tools.matmul(ih, V) + bc  # 之后以h作为xh作为跨层输入
            # self.gruParams[layer]['xh'] = xh
            xh = h
            ixh = ih
            # self.lstmParams[layer]['xh'] = xh
            # self.lstmParams[layer]['ixh'] = ixh

            # if layer == L -1: # 最后一层，做合并
            #     # U = self.gruParams[layer]['U']
            #     # V = self.gruParams[layer]['V']
            #     # bc = self.gruParams[layer]['bc']
            #
            #     # xh_ori = Tools.matmul(h, U) + Tools.matmul(ih, V) + bc
            #     # xh_final = xh_ori
            #     # xh_final =ReLU.activate( xh_ori )
            #     # xh_final =np.tanh( xh_ori )
            #     # xh_final =Tools.sigmoid( xh_ori )
            #
            #     xh_final = np.concatenate((xh, ixh), axis=2)  # 在H维度上做拼接
            #
            #     self.gruParams[layer]['xh_final'] = xh_final
                # self.gruParams[layer]['xh_ori'] = xh_ori
        # ixh = ixh[:,::-1,:] # 左向结果逆序
        xh_final = np.concatenate((xh, ixh), axis=2)  # 在H维度上做拼接
        self.gruParams[layer]['xh_final'] = xh_final
        return xh_final
        # return ixh

    def gru_backward(self, dh_all):
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        diWzx, diWzh, dibz, diWax, diWar, diba = None, None, None, None, None, None
        dU, dV, dbc = None, None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        N, T, H_time_2 = dh_all.shape #得到的误差是batch *T* 2H
        H = int(H_time_2 / 2)
        # N, T, H = dh_all.shape
        x, _, _, _, _, _, _, _, _, _ = self.gruParams[0]['cache'][0]
        D = x.shape[1]

        # # dh_prevl_all = dh_all
        # # 保存各层dwh,dwx,和db
        # dh_t_all = dh_all
        # U = self.gruParams[self.layersNum - 1]['U']
        # V = self.gruParams[self.layersNum - 1]['V']
        # xh_final =self.gruParams[self.layersNum - 1]['xh_final']
        # xh_ori =self.gruParams[self.layersNum - 1]['xh_ori']
        # dhih = dh_all
        # dhih = dh_all*Tools.bp4tanh(xh_final)
        # dhih = ReLU.bp(dh_all,xh_ori)
        # dhih = dh_all * Tools.bp4sigmoid(xh_final)

        # dh = Tools.matmul(dhih,U.T)
        # dih = Tools.matmul(dhih, V.T)

        dh = dh_all[:,:,0:H]
        # dh = dh_all
        # dih = dh_all[:,::-1,:] # 梯度反向
        # dih = dh_all # 梯度不反向

        # dih_2 = dh_all  # 梯度不反向
        # dih = dh_all
        dih = dh_all[:,:,H:2*H]

        dweights = []

        for layer in range(self.layersNum - 1, -1, -1):

            # U = self.gruParams[layer]['U']
            # V = self.gruParams[layer]['V']
            # h = self.gruParams[layer]['h']
            # ih = self.gruParams[layer]['ih']

            # dh_prevl = Tools.matmul(dh_t_all, U.T)
            # dih_prevl = Tools.matmul(dh_t_all, V.T)

            dh_prevl = dh
            dih_prevl = dih

            # dih_prevl_2 = dih_2 # 不反向

            DH = D if layer == 0 else H

            # 右向 得到前向传播保存的cache数组
            cache = self.gruParams[layer]['cache']

            dx = np.zeros((N, T, DH))
            dWzx = np.zeros((DH, 2 * H))
            dWzh = np.zeros((H, 2 * H))
            dbz = np.zeros((2 * H))

            dWax = np.zeros((DH, H))
            dWar = np.zeros((H, H))
            dba = np.zeros((H))

            dprev_h = np.zeros((N, H))

            # 左向
            icache = self.gruParams[layer]['icache']

            dix = np.zeros((N, T, DH))
            diWzx = np.zeros((DH, 2 * H))
            diWzh = np.zeros((H, 2 * H))
            dibz = np.zeros((2 * H))

            diWax = np.zeros((DH, H))
            diWar = np.zeros((H, H))
            diba = np.zeros((H))

            dprev_ih = np.zeros((N, H))

            ##########左向，不逆序#############################
            # dix_2 = np.zeros((N, T, DH))
            # diWzx_2 = np.zeros((DH, 2 * H))
            # diWzh_2 = np.zeros((H, 2 * H))
            # dibz_2 = np.zeros((2 * H))
            #
            # diWax_2 = np.zeros((DH, H))
            # diWar_2 = np.zeros((H, H))
            # diba_2 = np.zeros((H))
            #
            # dprev_ih_2 = np.zeros((N, H))
            #######################################

            # 前馈合并参数
            # xht = (self.lstmParams[layer]['xh'])
            # dU = np.zeros((H, H))
            # dV = np.zeros((H, H))
            # dbc_final = np.zeros((H))

            for t in range(T - 1, -1, -1):
                # 右向
                dx[:, t, :], dprev_h, dWzx_t, dWzh_t, dbz_t, dWax_t, dWar_t, dba_t = self.gru_step_backward(
                    dh_prevl[:, t, :] + dprev_h,
                    cache[t])  # 注意此处的叠加
                dWzx += dWzx_t
                dWzh += dWzh_t
                dbz += dbz_t

                dWax += dWax_t
                dWar += dWar_t
                dba += dba_t

                # 左向
                # dix_2[:, T - 1 - t, :], dprev_ih_2, diWzx_t_2, diWzh_t_2, dibz_t_2, diWax_t_2, diWar_t_2, diba_t_2 = self.gru_step_backward(
                #     dih_prevl_2[:, T - 1 - t, :] + dprev_ih_2,
                #     icache[T - 1 - t])  # 注意此处的叠加 不逆序

                dix[:, T-1-t, :], dprev_ih, diWzx_t, diWzh_t, dibz_t, diWax_t, diWar_t, diba_t = self.gru_step_backward(
                    dih_prevl[:, T- 1 - t, :] + dprev_ih,
                    icache[t])  # 注意此处的叠加，逆序

                # dix[:, t, :], dprev_ih, diWzx_t, diWzh_t, dibz_t, diWax_t, diWar_t, diba_t = self.gru_step_backward(
                #     dih_prevl[:, t, :] + dprev_ih,
                #     icache[t])  # 注意此处的叠加，逆序

                diWzx += diWzx_t
                diWzh += diWzh_t
                dibz += dibz_t

                diWax += diWax_t
                diWar += diWar_t
                diba += diba_t

                # # 层前馈合并参数
                # dU += Tools.matmul(h[:, t, :].T, dh_all[:, t, :])
                # dV += Tools.matmul(ih[:, t, :].T, dh_all[:, t, :])
                # if layer == L-1: # 最上一层,计算dU和dV
                # # 层前馈合并参数
                #     dU += Tools.matmul(h[:, t, :].T, dh[:, t, :])
                #     dV += Tools.matmul(ih[:, t, :].T, dih[:, t, :])

            # 需不需要给梯度除以批量
            # dU = dU / N
            # dV = dV / N
            # dbc_final = np.sum(np.sum(dh_all, axis=0), axis=0)

            # dweight = (dWx, dWh, db, diWx, diWh, dib, dU, dV, dbc_final)
            # dweights.append(dweight)
            #

            # 本层得出的dx，作为下一层的prev_l
            # dh_prevl = dx
            # if layer == L - 1:  # 最上一层,计算dU和dV
            #     dbc_final = np.sum(np.sum(dh_all, axis=0), axis=0)

            dweight = (dWzx, dWzh, dbz, dWax, dWar, dba,
                       diWzx, diWzh, dibz, diWax, diWar, diba
                       #,dU, dV, dbc_final
                       )
            dweights.append(dweight)

            # # 本层得出的dx，作为下一层的误差输入
            # 本层得出的dx，作为下一层的误差输入
            # dh_t_all = dx + dix
            dh = dx
            dih = dix
            # dih_2 = dix_2

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # 返回x误差和各层参数误差
        # dih = dih[:,::-1,:] # 误差逆向
        # r_dih = dih[:,:,::-1] # dhi逆序
        # dh_t_all = dh + r_dih # 合并得到dx
        dh_t_all = dh + dih # 合并得到dx
        return dh_t_all, dweights
        # return dih, dweights

    def gru_step_forward(self, x, prev_h, Wzx, Wzh, bz, Wax, War, ba):
        """
        Forward pass for a single timestep of an LSTM.

        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.

        Note that a sigmoid() function has already been provided for you in this file.

        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wzx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, cache = None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.
        # 首层，x(N,T,D), 向上变成xh(N,T,H)
        # 首层 Wx(D,H),   向上变成Wxh(H,H)
        #############################################################################
        H = prev_h.shape[1]
        # z_hat, of shape(N,4H)
        z_hat = Tools.matmul(x, Wzx) + Tools.matmul(prev_h, Wzh) + bz

        # of shape(N,H)
        r = Tools.sigmoid(z_hat[:, :H])
        z = Tools.sigmoid(z_hat[:, H:2 * H])

        a = Tools.matmul(x, Wax) + Tools.matmul(r * prev_h, War) + ba

        next_h = prev_h * (1. - z) + z * np.tanh(a)

        cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return next_h, cache

    def gru_step_backward(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh*(tanha-prev_h)
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))

        dhat_a = Tools.matmul(da, War.T)
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)

        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)

        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)

        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_succ(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        # dz_hat_2 = dz*(z*(1.-z))
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))
        # dz_hat_1 = prev_h * Tools.matmul(dh*z*(1-tanha**2), War.T)*(r*(1.-r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        dx_2 = Tools.matmul(dz_hat_1, Wzx[:, :H].T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dh_prev_3 = Tools.matmul(dz_hat_1, Wzh[:, :H].T)
        dx_23 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        dx_3 = Tools.matmul(dz_hat_2, Wzx[:, H:2 * H].T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        dh_prev_4 = Tools.matmul(dz_hat_2, Wzh[:, H:2 * H].T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34  #dh_prev_3 + dh_prev_4
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_23 # +dx_3
        dx = dx_1 + dx_2 + dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v2(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha ** 2)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        # dz_hat_2 = dz*(z*(1.-z))
        dz_hat_2 = dh * (tanha - prev_h) * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))
        # dz_hat_1 = prev_h * Tools.matmul(dh*z*(1-tanha**2), War.T)*(r*(1.-r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dx_23 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_34  # dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_23  # +dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v1(self, dnext_h, cache):
        """
        Backward pass for a single timestep of an LSTM.

        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha * tanha)
        dh_prev_1 = dh * (1. - z)
        # dz = dh * (z+tanha)
        # dz = dh*tanha+1.-dh*(1.-z)*prev_h
        # dz = dh*tanha+1.-dh*prev_h
        dz = dh * (tanha - prev_h)
        dz_hat_2 = dz * (z * (1. - z))
        # dz_hat_2 = dz*(z_hat_H2*(1.-z_hat_H2))

        dhat_a = Tools.matmul(da, War.T)
        # dz_hat_2 = dhat_r * r
        dr = dhat_a * prev_h

        dx_1 = Tools.matmul(da, Wax.T)
        dh_prev_2 = dhat_a * r  # da* Tools.matmul(r,War.T)
        # dz_hat_1 = dh_prev_2 * (r * (1. - r))
        dz_hat_1 = dr * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        # dx_2 = Tools.matmul(dz_hat_2,Wzx.T)
        # dh_prev_3 = Tools.matmul(dz_hat,Wzh.T)
        # dh_prev_3 = Tools.matmul(dz_hat_2,Wzh.T)
        dx_2 = Tools.matmul(dz_hat, Wzx.T)

        # dx_3 = Tools.matmul(dz_hat_1,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat_1, Wzh.T)
        # dx_3 = Tools.matmul(dz_hat,Wzx.T)
        # dh_prev_4 =Tools.matmul(dz_hat, Wzh.T)

        # dh_prev_34 = np.hstack((dh_prev_3, dh_prev_4))
        # dh_prev_34 = Tools.matmul(dh_prev_34,Wzh.T)
        dh_prev_34 = Tools.matmul(dz_hat, Wzh.T)
        # dprev_h = dh_prev_1+dh_prev_2+dh_prev_34 * 2. #dh_prev_3 + dh_prev_4
        # dx = dx_1 + dx_2*2. # +dx_3
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_34  # dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2  # +dx_3

        dWax = Tools.matmul(x.T, da)
        dWar = Tools.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = Tools.matmul(x.T, dz_hat)
        dWzh = Tools.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba

    def gru_step_backward_v0(self, dnext_h, cache):
        """
        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N, D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1 = z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh * z * (1. - tanha * tanha)
        dh_prev_1 = dh * (1. - z)
        dz = dh * (z + tanha)
        dz_hat_2 = dz * (z * (1. - z))

        d13 = np.matmul(da, War.T)
        dr = d13 * prev_h
        dx_1 = np.matmul(da, Wax.T)
        dh_prev_2 = d13 * r
        dz_hat_1 = dh_prev_2 * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1, dz_hat_2))

        dh_prev_3 = np.matmul(dz_hat, Wzh.T)
        dx_2 = np.matmul(dz_hat, Wzx.T)
        dx_3 = np.matmul(dz_hat, Wzx.T)
        dh_prev_4 = np.matmul(dz_hat, Wzh.T)
        dprev_h = dh_prev_1 + dh_prev_2 + dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 + dx_3

        dWax = np.matmul(x.T, da)
        dWar = np.matmul((r * prev_h).T, da)
        dba = np.sum(da, axis=0)

        dWzx = np.matmul(x.T, dz_hat)
        dWzh = np.matmul(prev_h.T, dz_hat)
        dbz = np.sum(dz_hat, axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba