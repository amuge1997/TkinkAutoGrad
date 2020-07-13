import numpy as n
from ThinkAutoGrad.Tensor import tensor


# 未测试
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
















