import numpy as n
from .Tensor import tensor

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

    # 未测试
    # 似乎存在性能问题
    @staticmethod
    def cat(tensorList, dim):
        arr = n.concatenate([ts.arr for ts in tensorList], axis=dim)
        dimshapels = [ts.shape[dim] for ts in tensorList]

        def getGrad(gradl, params):
            i = params
            gradn = cTools.rcat(gradl, dim, dimshapels, i)
            return gradn

        subTensorls = [(ts, getGrad, i) for i, ts in enumerate(tensorList)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 似乎存在性能问题
    @staticmethod
    def rcat(arr, dim, dimshapels, dimneedi=None):
        if dimneedi is not None:
            start = 0
            for i in range(dimneedi):
                start += dimshapels[i]
            end = start + dimshapels[dimneedi]
            slils = [slice(None, None, None) for i in range(len(arr.shape))]
            slils[dim] = slice(start, end, None)
            rarr = arr[tuple(slils)]
            return rarr
        else:
            start = 0
            rls = [None for i in range(len(dimshapels))]
            slils = [slice(None, None, None) for i in range(len(arr.shape))]
            for i, dimshape in enumerate(dimshapels):
                sli = slice(start, start + dimshape, None)
                slils[dim] = sli
                start += dimshape
                rls[i] = (arr[tuple(slils)])
            return rls










