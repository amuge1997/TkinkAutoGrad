from .Tensor import tensor
from .cTools import cTools
import numpy as n,time


class Layer:
    def __init__(self):
        self.params = None

    def forward(self,*tsrX):
        raise Exception('未重写该方法!')

    def paramsInit(self):
        raise Exception('未重写该方法!')

    def setGradZeros(self):
        raise Exception('未重写该方法!')

    def getParams(self):
        return self.params

class Relu(Layer):
    def __init__(self):
        super().__init__()

    # 未测试，但已经应用
    def relu(self,tensorIns):
        arr = n.where(tensorIns.arr > 0.0, tensorIns.arr, 0)

        def getGrad(gradl):
            gradn = n.where(tensorIns.arr > 0.0, 1.0, 0.0) * gradl
            return gradn

        subTensorls = [(tensorIns, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    def forward(self,*tsrX):
        tsrX = tsrX[0]
        Y = self.relu(tsrX)
        return Y


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    # 未测试，但已经应用
    def tanh(self,tensorIns):
        arr = n.tanh(tensorIns.arr)

        def getGrad(gradl):
            gradn = 1 - n.tanh(tensorIns.arr) ** 2
            gradn = gradn * gradl
            return gradn

        subTensorls = [(tensorIns, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    def forward(self,*tsrX):
        tsrX = tsrX[0]
        one = tensor(n.ones(tsrX.shape))
        five = tensor(n.ones(tsrX.shape) * 0.5)
        # Y = one / ( one + cts.exp( - tsrX ) )

        # 使用以下形式代替原sigmoid解决上溢出问题
        Y = five * ( one + self.tanh( five * tsrX ) )

        return Y

class Dense(Layer):
    def __init__(self,in_features,out_features,paramsC):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.paramsC = paramsC
        self.paramsInit()

    def paramsInit(self):
        in_features = self.in_features
        out_features = self.out_features
        paramsC = self.paramsC

        W = tensor(n.random.randn(in_features, out_features) / (n.sqrt(in_features * out_features)) * paramsC)
        b = tensor(n.random.randn(1, out_features) / (n.sqrt(1 * out_features)) * paramsC)

        W.setNeedUpdateGrad()
        b.setNeedUpdateGrad()

        self.params = {
            'w': W,
            'b': b,
        }

    def setGradZeros(self):
        W,b = self.params['w'],self.params['b']
        W.setGradZero(),b.setGradZero()

    # 原tile方法虽然完整但存在严重效率问题，因此使用以下临时方法
    # 未测试
    def btile(self,tensorIns,nDims):
        arr = n.tile(tensorIns.arr,(nDims,1))
        def getGrad(gradl):
            gradn = n.sum(gradl,axis=0,keepdims=True)
            return gradn
        subTensorls = [(tensorIns, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    def forward(self,*tsrX):
        self.setGradZeros()
        W, b = self.params['w'], self.params['b']
        tsrX = tsrX[0]

        # B = b.tile((tsrX.shape[0], 1))

        # 原tile方法虽然完整但存在严重效率问题，因此使用以下临时方法
        B = self.btile(b,tsrX.shape[0])


        Y = tsrX @ W + B
        return Y






























