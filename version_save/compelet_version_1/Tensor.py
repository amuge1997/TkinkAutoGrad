import numpy as n
import functools





class tensor:

    def __init__(self, inArr, subTensorls=None):

        self.arr = n.array(inArr)  # 这里相当于进行了深拷贝
        self.subTensorls = None

        self.shape = self.arr.shape

        self.grad = 0
        if subTensorls is None:
            self.subTensorls = []
        else:
            self.subTensorls = subTensorls

    # 清空子节点
    def emptySub(self):
        self.subTensorls = []

    # 已测试
    def backward(self, grad):
        if grad.shape != self.shape:
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, grad.shape))
        self.grad += grad
        for v in self.subTensorls:
            ins, func = v[0], v[1]
            subGrad = func(grad)
            ins.backward(subGrad)

    # 已测试
    def __getitem__(self, item):
        arr = self.arr[item]

        def getGrad(gradl):
            gradn = n.zeros_like(self.arr)
            gradn[item] = gradl
            return gradn

        subTensorls = [(self, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __matmul__(self, other):
        arr = self.arr @ other.arr

        def getGrad1(gradl):
            gradn = gradl @ other.arr.T
            return gradn

        def getGrad2(gradl):
            gradn = self.arr.T @ gradl
            return gradn

        subTensorls = [(self, getGrad1), (other, getGrad2)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __add__(self, other):

        # 元素对应加法维度必须相等
        if self.shape != other.shape:
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, other.shape))

        arr = self.arr + other.arr

        def getGrad1(gradl):
            gradn = n.ones_like(self.arr) * gradl
            return gradn

        def getGrad2(gradl):
            gradn = n.ones_like(other.arr) * gradl
            return gradn

        subTensorls = [(self, getGrad1), (other, getGrad2)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __sub__(self, other):

        # 元素对应减法维度必须相等
        if self.shape != other.shape:
            raise Exception('Tensor shape error! self.shape:{} grad.shape:{}'.format(self.shape, other.shape))

        arr = self.arr - other.arr

        def getGrad1(gradl):
            gradn = n.ones_like(self.arr) * gradl
            return gradn

        def getGrad2(gradl):
            gradn = - n.ones_like(other.arr) * gradl
            return gradn

        subTensorls = [(self, getGrad1), (other, getGrad2)]
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

        subTensorls = [(self, getGrad1), (other, getGrad2)]
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

        subTensorls = [(self, getGrad1), (other, getGrad2)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def __neg__(self):
        arr = - self.arr

        def getGrad(gradl):
            gradn = - n.ones_like(arr) * gradl
            return gradn

        subTensorls = [(self, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor


    # 未测试
    def transpose(self):
        arr = self.arr.T

        def getGrad(gradl):
            gradn = gradl.T
            return gradn

        subTensorls = [(self, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 未发现问题
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

        subTensorls = [(self, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 已测试
    def reshape(self, *shape):
        arr = self.arr.reshape(shape)

        def getGrad(gradl):
            gradn = gradl.reshape(self.arr.shape)
            return gradn

        subTensorls = [(self, getGrad)]
        newTensor = tensor(arr, subTensorls)
        return newTensor

    # 无问题
    def setGradZero(self):
        self.grad = 0

    # 无问题
    def __str__(self):
        sr = str(self.arr)
        return sr










