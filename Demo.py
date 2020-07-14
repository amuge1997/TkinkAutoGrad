from ThinkAutoGrad.Tensor import tensor
from ThinkAutoGrad.Layer import Dense,Sigmoid,Relu
import numpy as n,time

adam_dc = {}
def adam(neun, lr, epoch):
    p1 = 0.9
    p2 = 0.999
    e = 1e-8

    if neun in adam_dc:
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


def Demo():

    from Data_Samples import X,Y,n_samples,X1,X2

    X = tensor(X)
    Y = tensor(Y)

    sig = Sigmoid()
    relu = Relu()
    neuron1 = Dense(in_features=2, out_features=5, paramsC=1.5)
    neuron2 = Dense(in_features=5, out_features=5, paramsC=1.5)
    neuron3 = Dense(in_features=5, out_features=1, paramsC=1.5)

    cont = tensor(   n.ones((1, 1)) / (2 * n_samples)   )

    lr = 1e-3
    dc = {
        'fd':0.0,
        'bd':0.0,
        'up':0.0
    }
    for i in range(5000):
        start = time.time()

        o1 = neuron1.forward(X)
        o2 = sig.forward(o1)
        o3 = neuron2.forward(o2)
        o4 = sig.forward(o3)
        o5 = neuron3.forward(o4)
        Y_ = sig.forward(o5)
        L = cont * ((Y - Y_).transpose() @ (Y - Y_))

        end = time.time()
        runTime = end - start
        dc['fd'] += runTime

        start = time.time()
        L.backward(n.ones((1, 1)))
        end = time.time()
        runTime = end - start
        dc['bd'] += runTime

        start = time.time()
        adam(neuron1, lr, i + 1)
        adam(neuron2, lr, i + 1)
        adam(neuron3, lr, i + 1)
        end = time.time()
        runTime = end - start
        dc['up'] += runTime

        if (i+1) % 500 == 0 or i == 0:
            print('epoch:',i+1)
            print(L)

    sumTime = dc['fd'] + dc['bd'] + dc['up']
    fdrate = dc['fd'] / sumTime
    bdrate = dc['bd'] / sumTime
    uprate = dc['up'] / sumTime

    print()
    print('前向运行时间: {}s {}%'.format(round(dc['fd'], 2), round(fdrate * 100, 2)))
    print('反向运行时间: {}s {}%'.format(round(dc['bd'], 2), round(bdrate * 100, 2)))
    print('更新运行时间: {}s {}%'.format(round(dc['up'], 2), round(uprate * 100, 2)))
    print('总运行时间为: {}s'.format(round(sumTime, 2)))



    import matplotlib.pyplot as p
    import matplotlib as mpl
    # 网格区域绘制
    x = n.arange(-5, 5, 0.05)
    y = n.arange(-5, 5, 0.05)
    xx, yy = n.meshgrid(x, y)
    megX = n.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

    tensormegX = tensor(megX)

    o1 = neuron1.forward(tensormegX)
    o2 = sig.forward(o1)
    o3 = neuron2.forward(o2)
    o4 = sig.forward(o3)
    o5 = neuron3.forward(o4)
    predY = sig.forward(o5)

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
    Demo()


















