from keras import Sequential, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM
from keras.layers import Input, Dense, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizer_v2.adam import Adam
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import schedules
from tensorflow import keras
from scipy import signal
import scipy.io as scio
import tensorflow as tf
from sklearn.model_selection import train_test_split

a1=0
a2=0
a3=0
s1=0
s2=0
s3=0

features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据xx.csv",delimiter=",",usecols=[0,1,2])
labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据yy.csv",delimiter=",",usecols=[0])

features,a=np.array(np.array_split(features, 2))
labels,b=np.array(np.array_split(labels, 2))
print("features_shape:",features.shape,"labels_shape",labels.shape)
# 创建陷波滤波器
notch_freq = 50  # 陷波频率
Q = 30  # 品质因数

b, a = signal.iirnotch(notch_freq, Q, fs=100)

# 应用滤波器到信号
filtered_signal = signal.lfilter(b, a, features)
#___________________________________________________________start

# 数据分割
def segment_data(data, window_size):
    segments = []
    num_segments = len(data) // window_size
    for i in range(num_segments):
        segment = data[i * window_size: (i + 1) * window_size]
        segments.append(segment)
    return segments

# 特征提取 - 均方根 (RMS)
def calculate_rms(segment):
    rms = np.sqrt(np.mean(segment ** 2, axis=0))
    return rms

# 特征提取 - 过零率 (ZC)
def calculate_zc(segment):
    zc = np.sum(np.abs(np.diff(np.sign(segment))), axis=0) / (2 * len(segment))
    return zc
# 特征提取 - 绝对均值(MAV)
def calculate_mav(segment):
    mav = np.mean(np.abs(segment), axis=0)
    return mav

# 特征提取 - 示波指数（SSI）
def calculate_ssi(segment):
    ssi = np.sqrt(np.sum(np.abs(segment) ** 2, axis=0))
    return ssi

# 特征提取 - 平均功率谱密度（APSD）
def calculate_apsd(segment, sample_rate=100):
    freqs, psd = signal.welch(segment, fs=sample_rate)
    apsd = np.mean(psd, axis=0)
    return apsd

# 数据分割
window_size = 100  # 窗口大小，单位为毫秒
segments = segment_data(filtered_signal, window_size)

# 特征提取
rms_features = []
zc_features = []
for segment in segments:
    rms_segment = []
    zc_segment = []
    for i in range(segment.shape[1]):
        rms = calculate_rms(segment[:, i])
        zc = calculate_zc(segment[:, i])
        rms_segment.append(rms)
        zc_segment.append(zc)
    rms_features.append(rms_segment)
    zc_features.append(zc_segment)

features=np.hstack((rms_features,zc_features))
x_train=features
#___________________________________________________________
# x_train=features.reshape(len(features)//100,100,9)
#转换成百分比
for i in range(len(labels)):
    labels[i]=format((labels[i]/255)*100,'.2f')
y_train=labels
#print("feature:",x_train.shape)
#正则化，不行
#scaler = MinMaxScaler()
# for i in range(len(x_train)):
#     x_train[i] = scaler.fit_transform(x_train[i])


#平滑处理,感觉效果不太好
# x_train=np.round(savgol_filter(x_train, 5, 3, mode= 'nearest'),2)
# y_train=np.round(savgol_filter(y_train, 5, 3, mode= 'nearest'),2)
#归一化，将数据归类进0，1之间，显著加快拟合速度
# for i in range(len(x_train)):
#     mm = MinMaxScaler(feature_range=(0,1))
#     x_train[i]=mm.fit_transform(x_train[i])
#print(x_train[0:10])
# for i in range(len(labels)):
#     if labels[i]>=0 and labels[i]<45:
#         labels[i]=0
#     elif labels[i]>=45 and labels[i]<75:
#         labels[i] = 1
#     elif labels[i] >= 75 and labels[i] < 105:
#         labels[i] = 2
#     elif labels[i]>=105 and labels[i]<135:
#         labels[i] = 3
#可以改，现在这个是一行一个标签取全部数据，还可以试试50个取平均值然后取标签，可以只用后面6列作为数据集，


#____________________________________________________________________start
# 数据处理
nu = 4  # 输入延迟

x_processed = []
y_processed = []

for i in range(nu, len(x_train)):
    x_window = x_train[i-nu:i]
    y_window = y_train[i]
    x_processed.append(x_window)
    y_processed.append(y_window)

x_processed = np.array(x_processed)
y_processed = np.array(y_processed)
#
# # 数据归一化
# scaler = MinMaxScaler()
# x_shape = x_processed.shape
# x_processed = scaler.fit_transform(x_processed.reshape(-1, x_shape[-1])).reshape(x_shape)
# y_shape = y_processed.shape
# y_processed = scaler.fit_transform(y_processed.reshape(-1, y_shape[-1])).reshape(y_shape)


#_______________________________________________________________
# x_train=x_processed
#y_train=y_processed
# #建立模型
# model = Sequential()
# model.add(LSTM(300, input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True,kernel_regularizer=regularizers.l2(0.01)))
# model.add(keras.layers.BatchNormalization()),
# model.add(LSTM(300))
# model.add(keras.layers.BatchNormalization()),
# model.add(Dense(1))# 可能要用激活函数

#
#
# #编译模型
# lr_schedule = schedules.ExponentialDecay(
#     initial_learning_rate=0.01,
#     decay_steps=1000,
#     decay_rate=0.96,
#     staircase=True
# )
# optimizer = Adam(learning_rate=lr_schedule)
#
# model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
# # 打印模型结构
# model.summary()
# # 训练模型
# # print(x_train[0])
# # print(y_train[0:5])
# print("x_train",x_train.shape,"y_train",y_train.shape)
# history=model.fit(x_train, y_train, epochs=200, batch_size=128, validation_split=0.7)

#_____________________________________________________________start
# 数据划分
x_train, x_temp, y_train, y_temp = train_test_split(x_processed, y_processed, test_size=0.3, random_state=42)#训练数据，测试数据，训练标签，测试标签
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(nu, 6)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.RepeatVector(1),
    tf.keras.layers.LSTM(10, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
])
# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanSquaredError())
print(x_train[0:5])
# 训练模型
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, batch_size=128)


# 在测试集上评估模型
test_loss = model.evaluate(x_test, y_test)

# 打印测试集上的损失
print("Test Loss:", test_loss)

#画个图
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()
model.save(r'C:\Users\xsz\Desktop\pytorch学习\假肢3.0\model.h5')
# # 进行预测
# predictions = model.predict(x_test)
# # 打印预测结果
# print(predictions.shape)

# from keras import Sequential, regularizers
# from keras.layers import Embedding, LSTM
# from keras.layers import Input, Dense, Activation, Dropout
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.preprocessing import MinMaxScaler
#
# # from keras.optimizers import Adam           # tensorflow GPU 版2.10.0要用从这个地方引用Adam
# from keras.optimizer_v2.adam import Adam    # tensorflow CPU 版2.6.0要用从这个地方引用Adam
#
# # from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.optimizers import schedules
#
# a1=0
# a2=0
# a3=0
# s1=0
# s2=0
# s3=0
#
# features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据xx.csv",delimiter=",",usecols=[0,1,2,3,4,5,6,7,8])
# labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据yy.csv",delimiter=",",usecols=[0])
#
# features,a=np.array(np.array_split(features, 2))
# labels,b=np.array(np.array_split(labels, 2))
#
# print("feature:",features.shape)
# x_train=features.reshape(4425,100,9)
#
# print("x_train[0] ",x_train[0])
# print("x_train[1] ",x_train[1])
#
# # 归一化，不行
# # 找到最大值和最小值
# min_val = np.min(x_train)
# max_val = np.max(x_train)
#
# # 归一化二维数组的值到 0 到 1 之间
# normalized_array = (x_train - min_val) / (max_val - min_val)
#
# # 打印最大值、最小值和归一化后的数组
# print("最大值：", max_val)
# print("最小值：", min_val)
# #x_train=normalized_array
# print("归一化后的数组：")
# print(x_train.shape)
#
# print("x_train[0] ",x_train[0])
# print("x_train[1] ",x_train[1])
#
# y_train=labels
# print("y_train[0] ",y_train[0])
# print("y_train[1] ",y_train[1])
#
# min_val = np.min(y_train)
# max_val = np.max(y_train)
#
# # 归一化二维数组的值到 0 到 1 之间
# normalized_array = (y_train - min_val) / (max_val - min_val)
#
# # 打印最大值、最小值和归一化后的数组
# print("最大值：", max_val)
# print("最小值：", min_val)
# #y_train=normalized_array
# print("归一化后的数组：")
# print(y_train.shape)
#
# print("y_train[0] ",y_train[0])
# print("y_train[1] ",y_train[1])
#
#
#
# print("形状:x_train",x_train.shape,"y_train:",y_train.shape)
# print("x_train.shape[1] ",x_train.shape[1])
# print("x_train.shape[2] ",x_train.shape[2])
#
# model: Sequential = Sequential()
# model.add(LSTM(300, input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True,kernel_regularizer=regularizers.l2(0.001)))
# Dropout(0.5)
# model.add(LSTM(300))
# Dropout(0.5)
# model.add(Dense(1))# 可能要用激活函数
#
# #编译模型
# lr_schedule = schedules.ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=10000,
#     decay_rate=0.96,
#     staircase=True
# )
# optimizer = Adam(learning_rate=lr_schedule)
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# # 打印模型结构
# model.summary()
# # 训练模型
# history=model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.7)    # 取训练集的70%为测试集，这里是按顺序切，并不随机打散
#
# #画个图
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model train vs validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','validation'], loc='upper right')
# plt.show()
#
# model.save(r'C:\Users\xsz\Desktop\pytorch学习\假肢3.0\model.h5')
# # # 进行预测
# # predictions = model.predict(x_test)
# # # 打印预测结果
# # print(predictions.shape)