import numpy as n
from ThinkAutoGrad.Tensor import tensor

class cTools:

    # 未测试
    @staticmethod
    def exp(tensorIns):
        arr = n.exp(tensorIns.arr)

        def getGrad(gradl):

            # 元素对应减法维度必须相等
            # if tensorIns.arr.shape != gradl.shape:
            #     raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(tensorIns.arr.shape, gradl.shape))

            gradn = n.exp(tensorIns.arr) * gradl
            return gradn

        subTensorls = [(tensorIns, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 未测试
    @staticmethod
    def relu(tensorIns):
        arr = n.where(tensorIns.arr > 0.0, tensorIns.arr, 0)

        def getGrad(gradl):
            gradn = n.where(tensorIns.arr > 0.0, 1.0, 0.0) * gradl
            return gradn

        subTensorls = [(tensorIns, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    @staticmethod
    # 未测试
    def tanh(tensorIns):
        arr = n.tanh(tensorIns.arr)

        def getGrad(gradl):
            gradn = 1 - n.tanh(tensorIns.arr) ** 2
            gradn = gradn * gradl
            return gradn

        subTensorls = [(tensorIns, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor














