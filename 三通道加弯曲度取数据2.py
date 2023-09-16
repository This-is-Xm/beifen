import serial
import torch
import torch.nn as nn
import numpy as np
import time
import csv

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
    data_2d_list = []
    with open('finger1.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')# 创建一个二维列表来存储数据
        while True:
            size = ser.inWaiting()
            if size != 0:
                response = ser.read(size)
                data_buf += response
                while True:
                    end_index = bytes(data_buf).find(b"\xFF\x0D")
                    # print("len",len(data_buf))#每次读取数据可能比较少，所以需要读好几次才能读满53（50有效数据+2标志位+标签）
                    # if(len(data_buf)==53):#，然后，由于下方那个注释的理由，所以第一个数值为上一个数据的标签，即（1标签+50有效数据+2标志位）就有了53个数，此时标志位索引为52
                    #     print(data_buf)#所以data_buf[end_index + 2]会报错超出索引，所以这里条件必须为大于53
                    if end_index >= 0 and len(data_buf) > 53:
                        data = data_buf[1:end_index]
                        data_tag = data_buf[end_index + 2]  # 获取标签
                        data_buf = data_buf[end_index + 2:]#如果一次接收了一个通道以上的数据，后移，读下一个通道，因为每次取data_buf都忽略掉第一个y(即上一个的标签)，所以这里后移保留了第一个为y，防止之后每次读取都少第一个数
                        print("data",len(data),"data_tag",data_tag,"data_buf",len(data_buf))
                        if len(data) < 50:  # 修改为输入特征数量
                            data_mean = sum(data) // len(data) if len(data) > 0 else 0
                            data += [data_mean] * (50 - len(data))  # 修改为输入特征数量
                        # elif len(data) > 50:  # 修改为输入特征数量
                        #     data = data[-50:]  # 修改为输入特征数量
                        writer.writerow([data_tag] + data)
                    else:
                        break
# data [4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 3, 4, 3, 3, 4, 3, 4, 3, 4, 3, 3, 3, 4, 3, 4, 4, 3, 4, 4, 3, 3] data_tag 19 data_buf 1
# len 1
# len 20
# len 33
# len 53
# # [19, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 4, 4, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 5, 3, 3, 3, 3, 3, 3, 3, 4, 255, 13]255


if __name__ == '__main__':
    serial_open()
    try:
        Receive()
    except KeyboardInterrupt:
        serial_close()
