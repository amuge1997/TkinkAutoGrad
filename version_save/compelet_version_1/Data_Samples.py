import numpy as n


def data1():
    n_samples = 5
    X1 = n.random.normal(0, 0.5, (n_samples, 2)) + 2
    X2 = n.random.normal(0, 0.5, (n_samples, 2)) - 2
    X = n.concatenate((X1, X2))

    Y1 = n.zeros((n_samples, 1))
    Y2 = n.ones((n_samples, 1))
    Y = n.concatenate((Y1, Y2))


    return n_samples,X,Y,X1,X2


def data3():
    n_sample = 50
    n_samples = n_sample * 2

    offset = 1.3
    p11 = n.random.normal(0, 0.5, (n_sample, 2)) + offset

    p12 = n.random.normal(0, 0.5, (n_sample, 2)) - offset

    p21 = n.random.normal(0, 0.5, (n_sample, 2)) + offset
    p21[:, 1] -= 2 * offset

    p22 = n.random.normal(0, 0.5, (n_sample, 2)) - offset
    p22[:, 1] += 2 * offset

    inputX = n.concatenate((p11, p12, p21, p22), axis=0)

    l1 = n.zeros((n_samples, 1))

    l2 = n.ones((n_samples, 1))

    predY = n.concatenate((l1, l2), axis=0)

    X = inputX
    Y = predY
    X1 = X[0:n_samples]
    X2 = X[n_samples:]
    return n_samples,X,Y,X1,X2


def data4():
    n_samples = 100

    X11 = n.linspace(-n.pi,n.pi,n_samples).reshape(-1,1)
    X12 = 2 * n.sin(X11) + 1
    noise = n.random.normal(0,0.1,(n_samples,2))
    X11 += noise[:, 0:1]
    X12 += noise[:, 1:2]
    X1 = n.concatenate((X11,X12),axis=1)

    X21 = n.linspace(-n.pi, n.pi, n_samples).reshape(-1, 1)
    X22 = 2 * n.sin(X21) - 1
    noise = n.random.normal(0, 0.1, (n_samples, 2))
    X21 += noise[:, 0:1]
    X22 += noise[:, 1:2]
    X2 = n.concatenate((X21, X22), axis=1)

    X = n.concatenate((X1,X2),axis=0)

    Y1 = n.zeros((n_samples,1))
    Y2 = n.ones((n_samples,1))
    Y = n.concatenate((Y1,Y2))

    # p.scatter(X11, X12)
    # p.scatter(X21, X22)
    # p.grid()
    # p.show()

    return n_samples,X,Y,X1,X2

# n_samples, X, Y, X1, X2 = data4()

n_samples, X, Y, X1, X2 = data4()






