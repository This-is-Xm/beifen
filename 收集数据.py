# import serial
# import pandas as pd
# import csv
#
# # set serial port initialized parameters
# com = serial.Serial(
#     port='COM5',
#     baudrate=115200,
#     bytesize=serial.EIGHTBITS,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
# )
# store=[]#临时存储
# x_train=[]#训练集
# read=0#判断是否读取
# num=0#判断读取多少行数据
# time=0
# while True:
#     rxd = com.read(1)
#     if rxd.hex()=='dd':
#         read=1
#         time+=1
#     if read and time==1:
#         store.append(rxd.hex())
#         num+=1
#         if num==50:
#             x_train.append(store)
#             read=0
#             a=[i for i in x_train]
#             dataframe = pd.DataFrame({'通道一': a})
#             dataframe.to_csv(r"训练集.csv", sep=',')
#     if read and time==2:
#         store.append(rxd.hex())
#         num+=1
#         if num==50:
#             x_train.append(store)
#             read=0
#             a=[i for i in x_train]
#             dataframe = pd.DataFrame({'通道二': a})
#             dataframe.to_csv(r"训练集.csv", sep=',')
#
#     if read and time==3:
#         time=0
#         store.append(rxd.hex())
#         num+=1
#         if num==50:
#             x_train.append(store)
#             read=0
#             a=[i for i in x_train]
#             dataframe = pd.DataFrame({'通道三': a})
#             dataframe.to_csv(r"训练集.csv", sep=',',mode='a',index=False,header=False)

#这个是看这个模型正则哪个效果最好
from sklearn.datasets import make_moons
from keras import Sequential
from keras.layers import Embedding, LSTM
from keras.layers import Input, Dense, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt

from keras import regularizers

plt.rcParams['figure.dpi'] = 150

# 生成虚拟分类样本
features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据xx.csv",delimiter=",",usecols=[0,1,2,3,4,5,6,7,8])
labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据yy.csv",delimiter=",",usecols=[0])
#按照feature格式改
print("feature",features.shape)
features=features.reshape(8846,100,9)
# 划分训练集和测试集
n_train = 20
trainX, testX = features[:n_train], features[n_train:40]
trainy, testy = labels[:n_train], labels[n_train:40]


# 网格搜索参数配置
values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

all_train, all_test = list(), list()

for param in values:
    # 定义模型
    model = Sequential()
    model.add(LSTM(300, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    Dropout(0.5)
    model.add(LSTM(300))
    Dropout(0.5)
    model.add(Dense(1))  # 可能要用激活函数

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    # 训练模型
    model.fit(trainX, trainy, epochs=40, verbose=0)

    # 评估模型
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Param: %f, Train: %.3f, Test: %.3f' % (param, train_acc, test_acc))
    all_train.append(train_acc)
    all_test.append(test_acc)

plt.semilogx(values, all_train, label='train', marker='o')  # 转换为10的幂显示
plt.semilogx(values, all_test, label='test', marker='o')
plt.legend()
plt.show()
