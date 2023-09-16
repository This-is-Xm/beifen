
from keras import Sequential, regularizers
from keras.layers import Embedding, LSTM
from keras.layers import Input, Dense, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.optimizers import Adam           # tensorflow GPU 版2.10.0要用从这个地方引用Adam
# from keras.optimizer_v2.adam import Adam    # tensorflow CPU 版2.6.0要用从这个地方引用Adam

# from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import schedules

a1=0
a2=0
a3=0
s1=0
s2=0
s3=0

features = np.loadtxt(r"F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\符合输入格式的数据xx.csv",delimiter=",",usecols=[0,1,2,3,4,5,6,7,8])
labels = np.loadtxt(r"F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\符合输入格式的数据yy.csv",delimiter=",",usecols=[0])


print("feature:",features.shape)
x_train=features.reshape(8850,100,9)

print("x_train[0] ",x_train[0])
print("x_train[1] ",x_train[1])

# 归一化，不行
# 找到最大值和最小值
min_val = np.min(x_train)
max_val = np.max(x_train)

# 归一化二维数组的值到 0 到 1 之间
normalized_array = (x_train - min_val) / (max_val - min_val)

# 打印最大值、最小值和归一化后的数组
print("最大值：", max_val)
print("最小值：", min_val)
x_train=normalized_array
print("归一化后的数组：")
print(x_train.shape)

print("x_train[0] ",x_train[0])
print("x_train[1] ",x_train[1])

y_train=labels
print("y_train[0] ",y_train[0])
print("y_train[1] ",y_train[1])

min_val = np.min(y_train)
max_val = np.max(y_train)

# 归一化二维数组的值到 0 到 1 之间
normalized_array = (y_train - min_val) / (max_val - min_val)

# 打印最大值、最小值和归一化后的数组
print("最大值：", max_val)
print("最小值：", min_val)
y_train=normalized_array
print("归一化后的数组：")
print(y_train.shape)

print("y_train[0] ",y_train[0])
print("y_train[1] ",y_train[1])



print("形状:x_train",x_train.shape,"y_train:",y_train.shape)
print("x_train.shape[1] ",x_train.shape[1])
print("x_train.shape[2] ",x_train.shape[2])

model: Sequential = Sequential()
model.add(LSTM(300, input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True,kernel_regularizer=regularizers.l2(0.001)))
Dropout(0.5)
model.add(LSTM(300))
Dropout(0.5)
model.add(Dense(1))# 可能要用激活函数

#编译模型
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# 打印模型结构
model.summary()
# 训练模型
history=model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.7)    # 取训练集的70%为测试集，这里是按顺序切，并不随机打散

#画个图
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()

model.save(r'F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\假肢3.0\假肢3.0\model.h5')
# # 进行预测
# predictions = model.predict(x_test)
# # 打印预测结果
# print(predictions.shape)