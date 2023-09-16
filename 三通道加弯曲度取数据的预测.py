import serial
import torch
import torch.nn as nn
import numpy as np
import time
import csv

# 串口相关的函数
from keras.models import load_model
from numpy import argmax

ser = serial.Serial('com5', 115200, timeout=0.5)

def serial_open():
    ser.port = 'com5'
    ser.baudrate = 115200
    ser.bytesize = 8
    ser.stopbits = 1
    ser.parity = "N"
    if ser.isOpen():
        print("串口打开成功！")
    else:
        print("串口打开失败！")

def serial_close():
    ser.close()
    if ser.isOpen():
        print("串口关闭失败！")
    else:
        print("串口关闭成功！")

def Receive():
    #变量
    data_buf = []
    data_2d_list = []
    num=0
    a_all = []
    s_all = []
    a = 0
    s = 0
    data_real=np.full((100,9), np.NaN)
    time=1#次数，用来判断是否是第一次收录100个数据
    #预测
    model = load_model(r"C:\Users\xsz\Desktop\pytorch学习\假肢3.0\model.h5")
    with open('finger1.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')# 创建一个二维列表来存储数据
        while True:
            size = ser.inWaiting()
            if size != 0:
                response = ser.read(size)
                data_buf += response
                #print("长度",len(data_buf))
                while True:
                    end_index = bytes(data_buf).find(b"\xFF\x0D")#这个每次取的是最左边的
                    # 有几次读取会反常的接收上千行数据，如果这个数据保留，会直接导致后面的计算全部延后
                    # 因为要循环处理这上千行的数据（里面有很多很多标志位），而处理的这段时间里，接口还在不断接收数据（个人认为）数据还会累加，
                    # 所以具体表现为，如果print("长度",len(data_buf))，你会看到这个长度逐渐减小（这是在循环处理），当长度小于五十三时，长度会猛然跳到上万行（这是第二次接收的数据）
                    # if time==0:
                    #     time+=1
                    #     response = ser.read(size)
                    #     data_buf = data_buf[end_index + 2:end_index + 2]#清空，第一个不要了
                    #     #print("长度2", len(data_buf))
                    #     break
                    if end_index >= 0 and len(data_buf) > 53:
                        data = data_buf[1:end_index]
                        data_tag = data_buf[end_index + 2]  # 获取标签
                        data_buf = data_buf[end_index + 2:]#如果一次接收了一个通道以上的数据，后移，读下一个通道，因为每次取data_buf都忽略掉第一个y(即上一个的标签)，所以这里后移保留了第一个为y，防止之后每次读取都少第一个数
                        print("data", len(data), "data_tag", data_tag, "data_buf", len(data_buf))#观察data_buf长度在这里看
                        if len(data) < 50:  # 修改为输入特征数量
                            data_mean = sum(data) // len(data) if len(data) > 0 else 0
                            data += [data_mean] * (50 - len(data))  # 修改为输入特征数量
                        elif len(data) > 50:  # 修改为输入特征数量
                            data = data[-50:]  # 修改为输入特征数量
                        #此处开始预处理，主要流程为，将前三个数据（即三个通道第一行数据）分别竖直然后拼接起来，放进data1中，data2同理，然后将两者垂直拼接起来，就构成了所需要的数据格式（100，3）
                        #第一次的数据会保留在ready数组中，计算出平均值后与第二次的数据分别水平连接，改变形状为（1，100，9），预测完毕后保留ready的最后100行数据，用作下一次的求平均值
                        num+=1#用作计数，6个一循环
                        data=np.array(data).reshape(-1, 1)
                        if num==1:#第一个通道第一个数据 ——》 |
                            data1=data
                        elif num<=3:#第第二三个通道第一个数据    ——》 | | |
                            data1=np.hstack((data1,data))#把第一行三个通道数据在这存一下（50,1）-》（50,3）
                        elif num==4:#第一个通道第二个数据 ——》 |
                            data2 = data
                        else:#第第二三个通道第二个数据  ——》 | | |
                            data2=np.hstack((data2,data))#把第二行三个通道数据在这存一下（50,1）-》(50,3）
                        if num==6:#存好了（50,3）-》（100.3）
                            data_all=np.vstack((data1,data2))#(100,3)  ——》 | | |
                            #print("data_all",data_all.shape)#         ——》 | | |
                            num=0
                            if time==1:#第一次保存的100个数据
                                ready=data_all.copy()
                                time+=1
                            else:
                                ready=np.vstack((ready,data_all))
                                #print("ready",ready.shape)#ready (200, 3)
                                for time in range(100, len(ready)):
                                    for m in range(3):
                                        for i in range(10):  # 取前十个样本
                                            a += ready[time - 10 + i][m]
                                        a = a / 10
                                        a_all.append(a)

                                        for i in range(100):  # 取前一百个样本
                                            s += ready[time - 100 + i][m]
                                        s = s / 100
                                        s_all.append(s)
                                    #print("这里",np.hstack((ready[time], a_all, s_all)),time)#这里 [ 6.  65. 31.  6.  70.  6  40.06  6.  74.62  35.3062] 100
                                    data_real[time-100]= np.hstack((ready[time], a_all, s_all))
                                    a_all = []
                                    s_all = []
                                ready=ready[-100:]#保留用作下一次求平均值
                                # print("data_real",data_real.reshape(1,100,9).shape)#(1,100,9)
                                predicted_labels = model.predict(np.array(data_real).reshape(1,100,9))
                                with open("3.0result.csv", "ab") as f:#预测角度结果保存
                                    np.savetxt(f, argmax(predicted_labels).reshape(1, -1),fmt='%.04f')
                                with open("3.0train.csv", "ab") as f:#训练数据保存
                                    np.savetxt(f, np.array(data_real).reshape(100,9),fmt='%.04f')
                                with open("3.0labels.csv", "ab") as f:#实际角度保存，三个通道各有一个y，但是保存角度是实时的，所以会覆盖，每次保存的是第三个通道的y（三个y都一样，不影响）
                                    np.savetxt(f, np.array(data_tag).reshape(1, -1),fmt='%.04f')
                                print("预测：",argmax(predicted_labels),"实际",data_tag)
                                #data_real=np.full((100,9), np.NaN)#刷新下
                                # writer.writerow([data_tag] + data)
                    else:
                        break
                    while True:#由于会出现一次接收很多数据，上方循环持续处理这些数据会导致判断延后，所以每次若接收两个动作以上的数据，去掉这些数据，只保留末尾部分用来和下次收取数据衔接
                        end_index = bytes(data_buf).find(b"\xFF\x0D")
                        if end_index>=0:
                            data_buf = data_buf[end_index + 2:]
                        else:
                            break

if __name__ == '__main__':
    serial_open()
    try:
        Receive()
    except KeyboardInterrupt:
        serial_close()
