import numpy as n
import functools,time

st = 0.0
reshapeTime = 0.0

# 需要注意，使用 ones() 而非 ones_like() 后者运行时间为前者 1.4 倍

class tensor:
    def __init__(self, inArr, subTensorls=None):

        self.arr = n.array(inArr)  # 这里相当于进行了深拷贝
        self.subTensorls = None
        self.needUpdateGrad = False

        self.shape = self.arr.shape

        self.grad = 0
        if subTensorls is None:
            self.subTensorls = []
        else:
            self.subTensorls = subTensorls

    # 开启更新梯度
    def setNeedUpdateGrad(self):
        self.needUpdateGrad = True
    # 关闭更新梯度
    def closeNeedUpdateGrad(self):
        self.needUpdateGrad = False

    # 清空子节点
    def emptySub(self):
        self.subTensorls = []

    # 已测试
    def backward(self, grad):

        if grad.shape != self.shape:
            # print(grad.shape,self.shape)
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, grad.shape))

        if self.needUpdateGrad:
            self.grad += grad

        for v in self.subTensorls:
            ins, func, params = v[0], v[1], v[2]
            if params is None:
                subGrad = func(grad)
            else:
                subGrad = func(grad,params)
            ins.backward(subGrad)

    # 已测试
    def __getitem__(self, item):
        arr = self.arr[item]

        def getGrad(gradl):
            start = time.time()

            gradn = n.zeros(self.shape)
            gradn[item] = gradl

            end = time.time()
            global st
            st += end-start

            return gradn

        subTensorls = [(self, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 进行了测试，应该无问题
    # 这里进行了转置修改
    def __matmul__(self, other):
        arr = self.arr @ other.arr

        # 默认最后两个维度为矩阵
        def getGrad1(gradl):
            temp = [i for i in range(len(other.shape))]
            temp[-2] = -1
            temp[-1] = -2
            gradn = gradl @ n.transpose(other.arr,temp)
            return gradn

        def getGrad2(gradl):
            temp = [i for i in range(len(self.shape))]
            temp[-2] = -1
            temp[-1] = -2
            gradn = n.transpose(self.arr,temp) @ gradl
            return gradn

        subTensorls = [(self, getGrad1, None), (other, getGrad2, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __add__(self, other):

        # 元素对应加法维度必须相等
        if self.shape != other.shape:
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, other.shape))

        arr = self.arr + other.arr

        def getGrad1(gradl):
            gradn = n.ones(self.arr.shape) * gradl
            return gradn

        def getGrad2(gradl):
            gradn = n.ones(other.arr.shape) * gradl
            return gradn

        subTensorls = [(self, getGrad1, None), (other, getGrad2, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __sub__(self, other):

        # 元素对应减法维度必须相等
        if self.shape != other.shape:
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, other.shape))

        arr = self.arr - other.arr

        def getGrad1(gradl):
            gradn = n.ones(self.arr.shape) * gradl
            return gradn

        def getGrad2(gradl):
            gradn = - n.ones(other.arr.shape) * gradl
            return gradn

        subTensorls = [(self, getGrad1, None), (other, getGrad2, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __mul__(self, other):

        # 元素对应乘法维度必须相等
        if self.shape != other.shape:
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, other.shape))

        arr = self.arr * other.arr

        def getGrad1(gradl):
            gradn = other.arr * gradl
            return gradn

        def getGrad2(gradl):
            gradn = self.arr * gradl
            return gradn

        subTensorls = [(self, getGrad1, None), (other, getGrad2, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __truediv__(self, other):
        # 元素对应乘法维度必须相等
        if self.shape != other.shape:
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, other.shape))

        arr = self.arr / other.arr

        def getGrad1(gradl):
            gradn = 1 / other.arr * gradl
            return gradn

        def getGrad2(gradl):
            gradn = -1 * self.arr / (other.arr * other.arr) * gradl
            return gradn

        subTensorls = [(self, getGrad1, None), (other, getGrad2, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __neg__(self):
        arr = - self.arr

        def getGrad(gradl):
            gradn = - n.ones(arr.shape) * gradl
            return gradn

        subTensorls = [(self, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor


    # 未测试
    # 直接 T 是有问题的，只有是二维数组时才能直接使用T，否则将导致混乱
    def transpose(self):
        arr = self.arr.T

        def getGrad(gradl):
            gradn = gradl.T
            return gradn

        subTensorls = [(self, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 未发现问题
    # 该方法存在严重效率问题
    def tile(self, shapeTup):
        arr = n.tile(self.arr, shapeTup)

        orishape = list(self.shape)
        orishape = [1] * (len(shapeTup) - len(orishape)) + orishape

        # 递归得到每个子数组的slice元组，使用该元组进行切片
        def walk(s, i, slils, ori, resls):
            # s 输入形式列表
            # i 输入形式列表的索引值
            if i >= len(s):
                # print(slils)
                return True, tuple(slils)

            for j in range(s[i]):
                slils.append(slice(j * orishape[i], (j + 1) * orishape[i], None))
                flg, slitup = walk(s, i + 1, slils, ori, resls)
                if flg:
                    resls.append(ori[slitup])
                slils.pop(-1)
            return False, resls

        def getGrad(grad1):
            # 递归后得到每个子数组

            flg, resls = walk(shapeTup, 0, [], grad1, [])
            gradn = functools.reduce(lambda x, y: x + y, resls)

            return gradn

        subTensorls = [(self, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def reshape(self, *shape):
        arr = self.arr.reshape(shape)

        def getGrad(gradl):
            start = time.time()

            gradn = gradl.reshape(self.arr.shape)

            end = time.time()
            global reshapeTime
            reshapeTime += end - start

            return gradn

        subTensorls = [(self, getGrad, None)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 无问题
    def setGradZero(self):
        self.grad = 0

    # 无问题
    def __str__(self):
        sr = str(self.arr)
        return sr










