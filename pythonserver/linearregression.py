# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def getLinearData(lamda):
    x1 = np.random.random(50) * 20 - 10
    x2 = np.random.random(50) * 7 - 2
    x3 = np.random.random(50) * 36 - 13
    x4 = np.random.random(50) * 9 - 3
    x = np.stack((x1, x2, x3, x4), axis=1)
    b = np.ones((50))
    x = np.c_[x, b]
    y = np.sum(x*lamda, axis=1) + np.random.random(50) * 50 - 25
    return x, y


def predict_y(x, k):
    return np.sum(x*k, axis=1)


def cal_loss(x, y, k):
    y_predict = np.sum(x*k, axis=1)
    loss = 0.5 * np.sum(np.square(y - y_predict))
    return loss


def mini_batch_GD():
    return


def BGD(x, y, k, n, s, loss):
    losslist = []
    for i in range(n):

        y_hat = predict_y(x, k)
        for j, dk in enumerate(k):
            xi = x[:, j]
            dk = dk + s * np.sum((y-y_hat) * xi, axis=0)
            k[j] = dk
        new_loss = cal_loss(x, y, k)
        losslist.append(new_loss)
        if new_loss < loss or (i > 10 and np.abs(new_loss - losslist[-2]) < 0.000001):
            return k, new_loss, losslist
    new_loss = cal_loss(x, y, k)
    return k, new_loss, losslist


def SGD(x, y, k, n, s, loss):
    losslist = []
    for i in range(n):

        for j, xi in enumerate(x):
            y_predict = np.sum(xi*k)
            k = k + s * (y[j] - y_predict) * xi
            new_loss = cal_loss(x, y, k)
            losslist.append(new_loss)

        if new_loss < loss :
            return k, new_loss, losslist
    new_loss = cal_loss(x, y, k)
    return k, new_loss, losslist


def linear_Regression(x, y, flag, n, r, loss):
    size = np.shape(x)[1]
    k = np.ones(size)
    if flag == True:
        k, loss, losslist = BGD(x, y, k, n, r, loss)
    else:
        k, loss, losslist = SGD(x, y, k, n, r, loss)
    i = np.arange(np.size(losslist))
    plt.figure()
    if flag == True:
        plt.plot(i, losslist, 'r-', label='BGD')
        plt.legend()
        plt.savefig('lossBGD.jpg')
    else:
        plt.plot(i, losslist, 'r-', label='SGD')
        plt.legend()
        plt.savefig('lossSGD.jpg')
    return k, loss

def linearServer(K, flag, num, r, loss):
    lamda = K  # a1, a2, a3, a4, b
    x, y = getLinearData(lamda)
    x_train, y_train = x[0:40, :], y[0:40]
    k, loss = linear_Regression(x_train, y_train, flag, num, r, loss)
    return k, loss

if __name__ == '__main__':
    kkk = [2, 5, 7, 1, -5]
    k, loss = linearServer(kkk, False, 100, 0.00005, 100)
    print(k, loss)































