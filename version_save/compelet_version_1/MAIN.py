
import numpy as n
from Tensor import tensor
import cTools


class Layer:
    def __init__(self):
        self.params = None
        self.Inpt = None
        self.Oupt = None
        self.type = None

    def gradsInit(self):
        pass

    def paramsInit(self):
        pass

    def forward(self,Inpt):
        raise Exception('Layer is None')

    def backward(self,Grad):
        raise Exception('Layer is None')

    def getParams(self):
        return self.params

class ActivationLayer(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'activation'

class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward(self,Inpt):

        # Inpt : tensor
        # Oupt : tensor

        # Inpt.emptySub()

        One1 = tensor(n.ones(Inpt.shape))
        One2 = tensor(n.ones(Inpt.shape))

        Oupt = One1 / (  One2 + cTools.exp(- Inpt)  )

        return Oupt


class NeuronLayer(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'neuron'

class Dense(NeuronLayer):
    def __init__(self,in_features,out_features,paramsC=1.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.paramsC = paramsC

        self.paramsInit()
        self.gradsInit()

    def paramsInit(self):
        in_features = self.in_features
        out_features = self.out_features
        paramsC = self.paramsC
        self.params = {
            'w': tensor(n.random.randn(in_features, out_features) / (n.sqrt(in_features * out_features)) * paramsC ),
            'b': tensor(n.random.randn(1, out_features) / ( n.sqrt(1 * out_features)) * paramsC ),
        }

    def resetParams(self):

        W = self.params['w']
        b = self.params['b']

        # 清空原来的连接
        W.emptySub()
        b.emptySub()

        # 清空梯度
        W.setGradZero()
        b.setGradZero()



    def forward(self,Inpt):

        self.Inpt = Inpt
        W = self.params['w']
        b = self.params['b']

        self.resetParams()

        B = b.tile((Inpt.shape[0],1))
        Oupt = Inpt @ W + B

        return Oupt

    def backward(self,Grad):
        pass


adam_dc = {}

def adam(neun, lr, epoch):
    p1 = 0.9
    p2 = 0.999
    e = 1e-8

    if neun in adam_dc:
        # print('yes')
        sw = adam_dc[neun]['sw']
        rw = adam_dc[neun]['rw']

        sb = adam_dc[neun]['sb']
        rb = adam_dc[neun]['rb']
    else:
        sw = 0
        rw = 0
        sb = 0
        rb = 0

    dw = neun.params['w'].grad
    db = neun.params['b'].grad


    def _adam(s,r,grads):
        s_save = p1 * s + (1 - p1) * grads
        r_save = p2 * r + (1 - p2) * grads ** 2

        s = s_save / (1 - p1 ** epoch)
        r = r_save / (1 - p2 ** epoch)

        ret = - lr * s / (n.sqrt(r) + e)

        return ret,s_save,r_save

    adm, sw, rw = _adam(sw, rw, dw)
    neun.params['w'].arr += adm

    adm, sb, rb = _adam(sb, rb, db)
    neun.params['b'].arr += adm

    if neun not in adam_dc:
        adam_dc[neun] = {
            'sw':sw,
            'rw':rw,
            'sb':sb,
            'rb':rb
        }





def version_1_test_of_linearR():

    from Data_Samples import X,Y,n_samples,X1,X2

    X = tensor(X)
    Y = tensor(Y)

    sig1 = Sigmoid()
    # sig2 = Sigmoid()
    # sig3 = Sigmoid()
    neuron1 = Dense(in_features=2, out_features=5, paramsC=1.5)
    neuron2 = Dense(in_features=5, out_features=5, paramsC=1.5)
    neuron3 = Dense(in_features=5, out_features=1, paramsC=1.5)

    # print(X1.shape,X2.shape,X.shape,Y.shape)
    # print( ((Y - Y_).transpose() @ (Y - Y_)).reshape(1)  )

    cont = tensor(n.ones((1, 1)) * (1 / (2 * n_samples)))

    lr = 1e-3
    for i in range(8000):

        o1 = neuron1.forward(X)
        o2 = sig1.forward(o1)

        o3 = neuron2.forward(o2)
        o4 = sig1.forward(o3)

        o5 = neuron3.forward(o4)
        Y_ = sig1.forward(o5)

        L = cont * ((Y - Y_).transpose() @ (Y - Y_))
        L.backward(n.ones((1, 1)))

        adam(neuron1, lr, i + 1)
        adam(neuron2, lr, i + 1)
        adam(neuron3, lr, i + 1)

        if (i+1) % 500 == 0 or i == 0:
            print('epoch:',i+1)
            print(L)

        # raise Exception

    o1 = neuron1.forward(X)
    o2 = sig1.forward(o1)
    o3 = neuron2.forward(o2)
    o4 = sig1.forward(o3)
    o5 = neuron3.forward(o4)
    Y_ = sig1.forward(o5)
    # print(Y_)

    Y_ = n.where(Y_.arr <= 0.5, 0, 1)
    # print(Y_)

    import matplotlib.pyplot as p
    import matplotlib as mpl
    # 网格区域绘制
    x = n.arange(-5, 5, 0.05)
    y = n.arange(-5, 5, 0.05)
    xx, yy = n.meshgrid(x, y)
    megX = n.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

    tensormegX = tensor(megX)

    o1 = neuron1.forward(tensormegX)
    o2 = sig1.forward(o1)
    o3 = neuron2.forward(o2)
    o4 = sig1.forward(o3)
    o5 = neuron3.forward(o4)
    predY = sig1.forward(o5)

    predY = predY.arr
    predY = n.where(predY <= 0.5, 0, 1)

    Z = predY

    cm_light = mpl.colors.ListedColormap(['green', 'yellow'])
    p.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap=cm_light)

    # 数据样本绘制
    p.scatter(X1[:, 0], X1[:, 1], c='red')
    p.scatter(X2[:, 0], X2[:, 1], c='blue')
    p.show()



if __name__ == '__main__':
    version_1_test_of_linearR()


















