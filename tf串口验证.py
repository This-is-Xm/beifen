import serial
import torch
import numpy as np
import time
from keras.models import load_model
from numpy import argmax

# 串口相关的函数
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
    data_buf = []
    data_zuizhong = []
    data_2d_list = []  # 创建一个二维列表来存储数据
    model = load_model(r"C:\Users\xsz\Desktop\pytorch学习\假肢3.0\3.0model171.h5")
    while True:
        size = ser.inWaiting()
        if size != 0:
            response = ser.read(size)
            data_buf += response
            while True:
                end_index = bytes(data_buf).find(b"\xFF\x0D")
                if end_index >= 0 and len(data_buf) > end_index + 1:
                    data = data_buf[1:end_index - 2]
                    data_tag = data_buf[end_index - 1]  # 获取标签
                    data_buf = data_buf[end_index + 2:]
                    if len(data) < 250:  # 修改为输入特征数量
                        data_mean = sum(data) // len(data) if len(data) > 0 else 0
                        data += [data_mean] * (250 - len(data))  # 修改为输入特征数量
                    elif len(data) > 250:  # 修改为输入特征数量
                        data = data[-250:]  # 修改为输入特征数量
                    c = sum(data)/250
                    #print(data)
                    data_2d_list.append(c)  # 将数据添加到二维列表中（使用切片来复制列表）
                    if len(data_2d_list) == 6:
                        # 对每行数据取平均值
                        #print("data_2d_list",data_2d_list)
                        averaged_data = np.array(data_2d_list)
                        with open("label.csv", "ab") as f:
                            np.savetxt(f, np.array(data_tag).reshape(1, -1))
                        with open("train.csv", "ab") as f:
                            np.savetxt(f, averaged_data.reshape(1, -1),fmt='%.04f')
                        predicted_labels=model.predict(averaged_data.reshape(1, -1))#预测
                        with open("result.csv", "ab") as f:
                            np.savetxt(f, argmax(predicted_labels).reshape(1, -1),fmt='%.04f')
                        #print("predicted_labels",predicted_labels)
                        #print(f"预测值： {predicted_labels},标签值：{data_tag}")
                        #print("predicted_labels:",predicted_labels)
                        data_2d_list = []  # 清空
                        data_zuizhong.append(argmax(predicted_labels))
                        if len(data_zuizhong) == 6:
                            print(data_zuizhong)
                            print(argmax(np.bincount(data_zuizhong)))
                            # predicted_labels_modo = mode(data_zuizhong)
                            # print(f"预测值：{predicted_labels_modo}")
                            data_zuizhong = []
                else:
                    break
            time.sleep(0.001)
def mode(lst):
    counter = {}
    max_count = 0
    mode_value = None
    for item in lst:
        item_int = int(item[0])
        counter[item_int] = counter.get(item_int, 0) + 1
        if counter[item_int] > max_count:
            max_count = counter[item_int]
            mode_value = item_int
    return mode_value
if __name__ == '__main__':
    serial_open()
    input_size = 6
    hidden_size = 64
    output_size = 6  # 修改为实际的类别数量
    learning_rate = 0.001
    batch_size = 24
    num_epochs = 100
    try:
        Receive()
    except KeyboardInterrupt:
        serial_close()
