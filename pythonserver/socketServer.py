#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import socket
import threading
import json
import linearregression
import numpy as np
import pandas as pd
import randomforest

def main():
    # 创建服务器套接字
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 7777
    # 将套接字与本地主机和端口绑定
    serversocket.bind((host, port))
    # 设置监听最大连接数
    serversocket.listen(50)
    # 获取本地服务器的连接信息
    myaddr = serversocket.getsockname()
    print("服务器地址:%s" % str(myaddr))
    # 循环等待接受客户端信息
    while True:
        # 获取一个客户端连接
        clientsocket, addr = serversocket.accept()
        print("连接地址:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket)  # 为每一个请求开启一个处理线程
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass
    serversocket.close()
    pass


class ServerThreading(threading.Thread):

    def __init__(self, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("开启线程.....")
        try:
            # 接受数据
            msg = ''
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # 解码
                msg += rec.decode(self._encoding)
                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                print("输出当前接收的数据", msg)

                if msg.strip().endswith('over'):
                    print("对消息进行划分，空格划分后的数据")
                    print(msg)
                    msg = msg[:-4]
                    break
            print("打印收到的msg消息数据: ", msg)
            # 解析json格式的数据
            parmjson = json.loads(msg)
            print("打印解析后的数据: ", parmjson)

            ###############
            resultparm =[]
            type = parmjson['type']
            if type == 'LINEAR_REGRESSION':
                print('--------------------------------接受参数------------------------------')
                K = [parmjson['k1'], parmjson['k2'], parmjson['k3'], parmjson['k4'], parmjson['b']]
                flag = parmjson['flag']
                num = parmjson['num']
                r = parmjson['r']
                loss = parmjson['loss']
                print('--------------------------------调用模型------------------------------')

                k, loss = linearregression.linearServer(K, flag, num, r, loss)

                resultparm = {'k1':k[0], 'k2':k[1], 'k3':k[2], 'k4':k[3], 'b':k[4], 'loss':loss}

            if type == 'RANDOM_FOREST':
                print('--------------------------------接受参数------------------------------')
                N = parmjson['N']
                max_depth = parmjson['max_depth']
                max_leaf_nodes = parmjson['max_leaf_nodes']
                print('--------------------------------调用模型------------------------------')
                accuracy = randomforest.randomForest(N, max_depth, max_leaf_nodes)
                resultparm = {'accuracy': accuracy}


            index = np.arange(len(resultparm))
            result = pd.Series(resultparm)

            result = resultparm

            print(" 返回的数据: ", result)

            sendmsg = json.dumps(result)

            # 发送数据
            self._socket.send(("%s" % sendmsg).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束.....")

        pass

    def __del__(self):
        pass

if __name__ == "__main__":
    main()


