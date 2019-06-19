#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import sklearn as sk
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import numpy as np

path = 'F:\WebStom\PageHome\image\\randomforest\\'

def makedata():
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    listmap = ListedColormap(['r', 'g', 'b', '#FF9900'])
    N = 400
    data1, y = make_blobs(n_samples=N, n_features=2, random_state=0, centers=4, cluster_std=[1.3, 0.9, 1.8, 2.1])
    # print(data1)
    # print(y)
    plt.figure(figsize=(8, 7))
    plt.scatter(data1[:, 0], data1[:, 1], c=y, cmap=listmap, s=20)
    plt.grid(True)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('随机生成用于模拟的数据集')
    plt.savefig('random_forest.jpg')
    return data1, y


def randomForest(N, max_depth, max_leaf_nodes):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x, y = makedata()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
    Forest = RandomForestClassifier(n_estimators=N, max_depth=max_depth, criterion='gini', max_leaf_nodes=max_leaf_nodes)
    Forest.fit(x_train, y_train)
    y_hat = Forest.predict(x_test)
    loss = y_hat == y_test
    accuracy = np.mean(loss)

    # 画图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围

    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)

    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF', '#660000'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b', '#FF9900'])
    y_hat = Forest.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(figsize=(8, 7))
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(" RandomForest 模型的分类结果")
    plt.grid(True)
    plt.savefig('F:\\WebStom\\PageHome\\image\\randomforest\\random_forest_map.jpg')
    return accuracy

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    N = 10
    max_depth = 5
    max_leaf_nodes = 5
    accuracy = randomForest(N, max_depth, max_leaf_nodes)
    print(accuracy)
