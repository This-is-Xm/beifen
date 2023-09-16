#目前进度：能跑，还需要：现在加载的还是一个人的数据，还需要将十个人的数据合并。并且预处理部分还没有实现，以及预测部分还需要拿全新的数据去实时预测。
import scipy.io as scio
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(suppress=True, threshold=sys.maxsize)

data=scio.loadmat(r'C:\Users\xsz\Desktop\数据集\s1\S1_E1_A1.mat')
# print(data.keys())#dict_keys(['__header__', '__version__', '__globals__', 'emg', 'acc', 'stimulus', 'glove', 'subject', 'exercise', 'repetition', 'restimulus', 'rerepetition', 'age', 'circumference', 'frequency', 'gender', 'height', 'weight', 'laterality', 'sensor'])
sampling_rate=int(data['frequency'])
emg_data = data['emg']
# print(data['emg'].shape)#(130267, 16)
# print(data['glove'].shape)#(130267, 22)

# 创建陷波滤波器
notch_freq = 50  # 陷波频率
Q = 30  # 品质因数

b, a = signal.iirnotch(notch_freq, Q, fs=sampling_rate)

# 应用滤波器到信号
filtered_signal = signal.lfilter(b, a, emg_data)

# 应用PCA进行降维
pca = PCA(n_components=6)  # 设置降维后的维度为6
reduced_data = pca.fit_transform(data['glove'])
#print("降维后的数据形状:", reduced_data.shape)  # (130267, 6)
# print("sEMG",data['emg'][400:420])
#print("data:",filtered_signal[400:500])
#print("labels:",reduced_data[0:100])

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
window_size = 200  # 窗口大小，单位为毫秒
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

#
# # 打印特征
# for i in range(len(rms_features)):
#     print(f"Segment {i+1}:")
#     for j in range(len(rms_features[i])):
#         print(f"Channel {j+1}: RMS = {rms_features[i][j]}, ZC = {zc_features[i][j]}")
rms_features=np.array(rms_features)
zc_features=np.array(zc_features)
features=np.hstack((rms_features,zc_features))
#print(features.shape)#(651, 32)
# 数据准备
x_data = features  # 输入数据 x(t)，32维特征 (RMS 和 ZC)
y_data = reduced_data  # 输出数据 y(t)，6维输出 (角位移)

# 数据处理
nu = 4  # 输入延迟

x_processed = []
y_processed = []

for i in range(nu, len(x_data)):
    x_window = x_data[i-nu:i]
    y_window = y_data[i]
    x_processed.append(x_window)
    y_processed.append(y_window)

x_processed = np.array(x_processed)
y_processed = np.array(y_processed)

# 数据归一化
scaler = MinMaxScaler()
x_shape = x_processed.shape
x_processed = scaler.fit_transform(x_processed.reshape(-1, x_shape[-1])).reshape(x_shape)
# y_shape = y_processed.shape
# y_processed = scaler.fit_transform(y_processed.reshape(-1, y_shape[-1])).reshape(y_shape)

print(y_processed[0:20])
# 获取最小值和最大值
min_label = scaler.data_min_
max_label = scaler.data_max_
# 数据划分
x_train, x_temp, y_train, y_temp = train_test_split(x_processed, y_processed, test_size=0.3, random_state=42)#训练数据，测试数据，训练标签，测试标签
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

print(x_train.shape)
# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(nu, 32)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.RepeatVector(1),
    tf.keras.layers.LSTM(10, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(6))
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanSquaredError())
print(y_train.shape)
# 训练模型
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64)

# 在测试集上评估模型
test_loss = model.evaluate(x_test, y_test)

# 打印测试集上的损失
print("Test Loss:", test_loss)

model.save('model.h5')
loaded_model = tf.keras.models.load_model('model.h5')
# print("先试试一个的",x_test[0])
result=loaded_model.predict(x_test)
print("结果",result[0:3])
print("实际上是",y_test[0:3])
# print("再试试多个的",x_test[0:3])
# result=loaded_model.predict(x_test[0:3])
# print("结果",result)
# print("实际上是",y_test[0:3])