import numpy as np
from keras.models import load_model
from numpy import argmax
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal

model = load_model(r"C:\Users\xsz\Desktop\pytorch学习\假肢3.0\model.h5")
features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据xx.csv",delimiter=",",usecols=[0,1,2])
labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据yy.csv",delimiter=",",usecols=[0])
# x_train=features.reshape(len(features)//100,100,9)

a,features=np.array(np.array_split(features, 2))
b,labels=np.array(np.array_split(labels, 2))
print("features_shape:",features.shape,"labels_shape",labels.shape)
# 创建陷波滤波器
notch_freq = 50  # 陷波频率
Q = 30  # 品质因数

b, a = signal.iirnotch(notch_freq, Q, fs=100)
# 应用滤波器到信号
filtered_signal = signal.lfilter(b, a, features)

#_______________________________________________
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

# # 数据归一化
# scaler = MinMaxScaler()
# x_shape = x_processed.shape
# x_processed = scaler.fit_transform(x_processed.reshape(-1, x_shape[-1])).reshape(x_shape)
# y_shape = y_processed.shape
# y_processed = scaler.fit_transform(y_processed.reshape(-1, y_shape[-1])).reshape(y_shape)

features=np.hstack((rms_features,zc_features))


# x_train=savgol_filter(x_train, 5, 3, mode= 'nearest')
# for i in range(len(x_train)):
#     mm = MinMaxScaler(feature_range=(0,10))
#     x_train[i]=mm.fit_transform(x_train[i])
# for i in range(len(labels)):
#     if labels[i]>=0 and labels[i]<45:
#         labels[i]=0
#     elif labels[i]>=45 and labels[i]<75:
#         labels[i] = 1
#     elif labels[i] >= 75 and labels[i] < 105:
#         labels[i] = 2
#     elif labels[i]>=105 and labels[i]<135:
# #         labels[i] = 3
# right=0
# for num in range(4):
#     for i in range(len(x_train)//4*num,len(x_train)//4*(num+1)):
#         predict = model.predict(x_train[i].reshape(1,x_train.shape[1],x_train.shape[2]))
#         if argmax(predict)==int(labels[i]):
#             right+=1
#     print(right,len(features)//4)
#     print("标签:",num,"正确率:",right*100//(len(x_train)//4),"%")
#     right=0


#之前预测结果全是0的情况，是因为输出只有一个数据，argmax()原本是用于独热转换标签的，因为只有一个，所以索引一直为0
# predict=[]
# alpha = 0.3  # 平滑系数
# previous_prediction = None
# n=10
# for num in range(len(labels[0:100])):
#     for i in range(n):  # 预测n个数据
#         result=model.predict(x_train[num].reshape(1, 4,6))# 进行实时预测，返回单个预测结果.原始数据就是3，论文数据就是9,另一个论文是6
#         if previous_prediction is None:
#             smoothed_prediction = result
#         else:
#             smoothed_prediction = alpha * result + (1 - alpha) * previous_prediction
#         previous_prediction = smoothed_prediction
#     predict.append(smoothed_prediction)
#     print("标签:",y_train[num],"预测:",predict[num][0])


# print(np.array(predict).reshape(-1).shape)
x_train, x_temp, y_train, y_temp = train_test_split(x_processed, y_processed, test_size=0.3, random_state=42)#训练数据，测试数据，训练标签，测试标签
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
#______________________________________
print(x_train[0:5])
predict=[]
for num in range(len(x_train[0:100])):
    result = model.predict(x_train[num].reshape(1,x_train.shape[1],x_train.shape[2]))
    predict.append(result)
    print("标签:", y_train[num], "预测:", result)
#画个图看看
plt.plot(np.array(predict).reshape(-1))
plt.plot(y_train[0:100])
plt.title('model train vs validation loss')
plt.ylabel('jiaodu')
plt.xlabel('epoch')
plt.legend(['predict','labels'], loc='upper right')
plt.show()







# model = load_model(r"C:\Users\xsz\Desktop\pytorch学习\假肢3.0\model.h5")
# features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据xx.csv",delimiter=",",usecols=[0,1,2,3,4,5,6,7,8])
# labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据yy.csv",delimiter=",",usecols=[0])
#
# a,features=np.array(np.array_split(features, 2))
# b,labels=np.array(np.array_split(labels, 2))
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
#
# #x_train=normalized_array
#
#
#
# y_train=labels
#
#
# min_val = np.min(y_train)
# max_val = np.max(y_train)
#
# # 归一化二维数组的值到 0 到 1 之间
# normalized_array = (y_train - min_val) / (max_val - min_val)
#
# # 打印最大值、最小值和归一化后的数组
# #y_train=normalized_array
#
# #预测
# predict=[]
# for num in range(len(x_train[0:300])):
#     result = model.predict(x_train[num].reshape(1,x_train.shape[1],x_train.shape[2]))
#     predict.append(result)
#     print("标签:", y_train[num], "预测:", result)
# #画个图看看
# plt.plot(np.array(predict).reshape(-1))
# plt.plot(y_train[0:300])
# plt.title('model train vs validation loss')
# plt.ylabel('jiaodu')
# plt.xlabel('epoch')
# plt.legend(['predict','labels'], loc='upper right')
# plt.show()

