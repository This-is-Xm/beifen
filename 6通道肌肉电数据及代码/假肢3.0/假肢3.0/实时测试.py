
from keras import Sequential, regularizers
from keras.layers import Embedding, LSTM
from keras.layers import Input, Dense, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler


model = load_model(r"F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\假肢3.0\假肢3.0\model.h5")
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

x_train=normalized_array



y_train=labels


min_val = np.min(y_train)
max_val = np.max(y_train)

# 归一化二维数组的值到 0 到 1 之间
normalized_array = (y_train - min_val) / (max_val - min_val)

# 打印最大值、最小值和归一化后的数组
y_train=normalized_array

#预测
predict=[]
for num in range(len(x_train[0:600])):
    # print("len(x_train[0:300]):", len(x_train[0:300]))    # 值恒为300
    # print("len(x_train[0:300]):", len(x_train[0:300]))    # 值恒为300
    # print("x_train.shape[1] ", x_train.shape[1])          # 值为100
    # print("x_train.shape[2] ", x_train.shape[2])          # 值为9
    x_predict=x_train[num].reshape(1, x_train.shape[1], x_train.shape[2])
    print("x_predict:", x_predict)

    result = model.predict(x_train[num].reshape(1,x_train.shape[1],x_train.shape[2]))
    print("result:", result)

    predict.append(result)
    print("num:", num)
    print("标签:", y_train[num], "预测:", result)
#画个图看看
plt.plot(np.array(predict).reshape(-1))
plt.plot(y_train[0:600])
plt.title('model train vs validation loss')
plt.ylabel('jiaodu')
plt.xlabel('epoch')
plt.legend(['predict','labels'], loc='upper right')
plt.show()

