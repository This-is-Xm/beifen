import joblib
import numpy as np
from keras.layers import Input, Dense, Activation, Dropout
from keras import Sequential
from keras.layers import Embedding, LSTM
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


features = np.loadtxt(open(r"C:\Users\xsz\Desktop\x.csv","rb"),delimiter=",",usecols=[0,1,2])
labels = np.loadtxt(open(r"C:\Users\xsz\Desktop\y2.csv","rb"),delimiter=",",usecols=[0])
# features, labels = joblib.load('/content/drive/My Drive/challenge/data.dat') #取数据#x这是linux取数据的，我先给注释起来
# print(len(features))
# print(len(labels))
features=features.reshape(10000,50,3)#x变形
labels=labels.reshape(10000,50,1)
print(features.shape)
print(labels.shape)
n_samples = len(features)  #打印数据的形状
# print(n_samples)
# for i in range(n_samples):
#   print(features[i].shape, labels[i].shape)
# for i in range(len(features)):
#   print(features[i].shape[0])
n_time_steps = min([features[i].shape[0] for i in range(len(features))]) #min值
# print(n_time_steps)
n_time_steps = max([features[i].shape[0] for i in range(len(features))]) #max值
# print(n_time_steps)
n_features = features[-1].shape[1] #样本数
n_labels = labels[-1].shape[1] #标签数
# print(n_samples, n_time_steps, n_features)
# print(n_labels)

#填充两个nan数组
features_extended = []
labels_extended = []
for i in range(n_samples):
  features_extended.append(np.full((n_time_steps, n_features), np.NaN))
  labels_extended.append(np.full((n_time_steps, n_labels), np.NaN))
#print(features_extended)
#print(labels_extended)
#features_extended[0]
#labels_extended[0]
# 不要为缺失的行（nan 值）插补平均值，而是使用最后一个非 nan 行;
# 这在直觉上似乎更合适，以避免特征或标签的非物理突然变化
#注释的代码也是处理nan数据的一种方式
#for i in range(n_samples):
#  features_extended[i] = np.where(np.isnan(features_extended[i]), np.ma.array(features_extended[i], mask=np.isnan(features_extended[i])).mean(axis=0), features_extended[i])

#for i in range(n_samples):
#  labels_extended[i] = np.where(np.isnan(labels_extended[i]), np.ma.array(labels_extended[i], mask=np.isnan(labels_extended[i])).mean(axis=0), labels_extended[i])

#填充features_extended = []和labels_extended = []
# (51,5000,30)
# (51,5000,6)
for i in range(n_samples):
  for j in range(features[i].shape[0]):
      features_extended[i][j,:] = features[i][j,:]
      labels_extended[i][j,:] = labels[i][j,:]
  for j in range(features[i].shape[0], features_extended[i].shape[0]):
    features_extended[i][j,:] = features[i][-1,:]
    labels_extended[i][j,:] = labels[i][-1,:]
# features_extended[0]
# features_extended[6]
# labels_extended[0]
# labels_extended[6]
X = np.array(features_extended)
#print(X.shape)  形状为(51, 5000, 30)
#X = np.vstack(X)
#print(X.shape)

# 规范化数据，使其特征值介于 0 和 1 之间：
# 除以最大可能特征值（1024 = 10 位分辨率）
# 还可以通过将分辨率降低到 4 字节来节省内存
X = X.astype('float32') / 1024.0 # 下面也会使用 StandardScaler，会在列车集上训练，更适合 NN 应用#转换为float形式/1024，使得得到一个在0到1范围内的浮点数数组 #x这个处理必须经过--------------------
#np.min(X), np.max(X), np.mean(X), np.std(X)  #最小值、最大值、平均值和标准差
y = np.array(labels_extended)
#print(y.shape)
#y = np.vstack(y)
#print(y.shape)

# 还可以安全地将标签的分辨率降低到 4 个字节
y = y.astype('float32')
#np.min(y), np.max(y), np.mean(y), np.std(y)
# print(X.shape)   (51, 5000, 30)
# print(y.shape)#   (51, 5000, 5)

#相同的拆分百分比：
# 训练/valid_test拆分 = 70%/30%，有效/测试拆分 = 50%/50%
# 因此，训练/有效/测试拆分 = 70%/15%/15%

# 也许不应该将它用于这样的时间序列数据，但下面的拆分代码会导致 NN 拟合出现问题
# X_train， X_valid_test， y_train， y_valid_test = train_test_split（X， y， test_size=0.30， random_state=42）
# X_valid， X_test， y_valid， y_test = train_test_split（X_valid_test， y_valid_test， test_size=0.50， random_state=42）

###

# 保留时间信息!!!

# 下面的拆分代码会导致 NN 拟合出现问题，所以不要使用它（可能只与 RNN 一起使用）
# 拆分时间序列数据时要小心：使用 train_test_split 会随机对数据进行重新排序
# 这里的简单代码在不重新排序任何数据的情况下执行相同的拆分
# 训练/验证/测试拆分 = 前 70% / 后 15% / 后 15%
#原文件是使用下面的方法
#print(n_samples)
n_samples = X.shape[0]
#print(n_samples)
n_samples_train = int(X.shape[0]*0.70)
#print(n_samples_train)

# x
# X_train（35，5000，30）训练数据集
# X_valid_test (16,5000,30)测试数据集
# y_train (35,5000,5)测试数据集
# y_valid_test (16,5000,5)测试标签

X_train = X[:n_samples_train]
X_valid_test = X[n_samples_train:]
y_train = y[:n_samples_train]
y_valid_test = y[n_samples_train:]

n_samples_valid_test = X_valid_test.shape[0]
#print(n_samples_valid_test)
n_samples_valid = int(X_valid_test.shape[0]*0.50)#x测试集的数据对半分,一半用于训练时看看loss，一半完全不参与训练，用于模型训练完毕之后进行预测
n_samples_test = int(X_valid_test.shape[0]*0.50)
#print(n_samples_valid)
#print(n_samples_test)
X_valid = X_valid_test[:int(X_valid_test.shape[0]*0.50)]
X_test = X_valid_test[int(X_valid_test.shape[0]*0.50):]
y_valid = y_valid_test[:int(X_valid_test.shape[0]*0.50)]
y_test = y_valid_test[int(X_valid_test.shape[0]*0.50):]

# 还可以用这个
#使用 shuffle=False 保留时间信息，
#X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.30, shuffle=False)
#X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.50, shuffle=False)
###  打印
# print(X.shape, y.shape)
# print(X_train.shape, y_train.shape)
# print(X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

# X arrays: convert to the type of 3d arrays expected by LSTM: X_array(data_lines, lag_steps, n_features_shifted)
#将x转换为lstm需要的形状
# first do what is normally done in autoregression: the present depends on some number of time steps n in the past:
# see, e.g., https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# y_vector(t) = f(X_matrix(t-n), y_vector(t-n), X_matrix(t-n+1), y_vector(t-n+1), ...,  X_matrix(t-n+1), y_vector(t-n+1), ..., X_matrix(t-1), y_vector(t-1), X_matrix(t-1), y_vector(t-1))
# here we take n = lag_steps = 1, so:
# y_vector(t) = f(X_matrix(t-1), y_vector(t-1))

# to start, forecast the present by looking just one lag step into the past

#转换为lstm所需的格式
lag_steps = 1
n_features_shifted = lag_steps*n_features#n_features_shifted = lag_steps*(n_features+n_labels)

X_train_shifted = np.full((n_samples_train, n_time_steps, n_features_shifted), np.NaN)# x三维对照组
y_train_shifted = np.full((n_samples_train, n_time_steps, n_labels), np.NaN)
for i in range(n_samples_train):
  # X_train_shifted[i] = np.hstack((X_train[i], y_train[i]))# x将每行的数据（30）和标签（5）对接,(35,5000,35)
  X_train_shifted[i]=X_train[i]
  y_train_shifted[i, 0:-1, :] = y_train[i, lag_steps:, :]# x移植，预测下一个角度值y(2,3,0)(1,2,3)------
#x预测这部分连接我删掉了
X_valid_shifted = np.full((n_samples_valid, n_time_steps, n_features_shifted), np.NaN)
y_valid_shifted = np.full((n_samples_valid, n_time_steps, n_labels), np.NaN)


for i in range(n_samples_valid):
   # X_valid_shifted[i] = np.hstack((X_valid_test[i], y_valid_test[i]))
   X_valid_shifted[i] = (X_valid_test[i])
   y_valid_shifted[i, 0:-1, :] = y_valid_test[i, lag_steps:, :]
print("我看看x内容:",X_train_shifted[0][:])
X_test_shifted = np.full((n_samples_test, n_time_steps, n_features_shifted), np.NaN)
y_test_shifted = np.full((n_samples_test, n_time_steps, n_labels), np.NaN)
for i in range(n_samples_test):
  # X_test_shifted[i] = np.hstack((X_test[i], y_test[i]))
  X_test_shifted[i]=X_test[i]
  y_test_shifted[i, 0:-1, :] = y_test[i, lag_steps:, :]

# print(X_train_shifted.shape, y_train_shifted.shape)
# print(X_valid_shifted.shape, y_valid_shifted.shape)
# print(X_test_shifted.shape, y_test_shifted.shape)
# print()
#
# print('X_train_shifted:')
# print(X_train_shifted)
# print()
#
# print('y_train_shifted:')
# print(y_train_shifted)
# print()
#
# print('y_valid_shifted:')
# print(y_valid_shifted)
# print()
#
# print('y_test_shifted:')
# print(y_test_shifted)
# print()


# 同样，不要为缺失的行（nan 值）插补平均值，而是使用最后一个非 nan 行
# 或者可以只删除行，但这会改变数据大小
# 希望有 5000 个时间步长，吃最后一行不会造成任何负面伪影
for i in range(n_samples_train):
  y_train_shifted[i, -1, :] = y_train_shifted[i, -2, :]

for i in range(n_samples_valid):
  y_valid_shifted[i, -1, :] = y_valid_shifted[i, -2, :]

for i in range(n_samples_test):
  y_test_shifted[i, -1, :] = y_test_shifted[i, -2, :]

# print(X_train_shifted.shape, y_train_shifted.shape)
# print(X_valid_shifted.shape, y_valid_shifted.shape)
# print(X_test_shifted.shape, y_test_shifted.shape)
# print()
#
# print('X_train_shifted:')
# print(X_train_shifted)
# print()
#
# print('y_train_shifted:')
# print(y_train_shifted)
# print()
#
# print('y_valid_shifted:')
# print(y_valid_shifted)
# print()
#
# print('y_test_shifted:')
# print(y_test_shifted)
# print()

#print('n_features_shifted: ', n_features_shifted)
n_features_shifted = X_train_shifted.shape[2]
#print('n_features_shifted: ', n_features_shifted)


scaler = StandardScaler()

#将3D数组X_train_shifted、y_train_shifted、X_valid_shifted和y_valid_shifted转换为2D数组，
#然后使用scaler对象对数据进行缩放。下面是将它们转换为2D数组的代码：

#print(X_train_shifted.shape)
X_train_shifted = np.vstack(X_train_shifted)# 将35组数据连成了一整个，从(35,5000,35)变成了（17500,35）行列

#print(X_train_shifted.shape)

#print(X_valid_shifted.shape)
X_valid_shifted = np.vstack(X_valid_shifted)
#print(X_valid_shifted.shape)

#print(X_test_shifted.shape)
X_test_shifted = np.vstack(X_test_shifted)
#print(X_test_shifted.shape)

#print()

# 注释中这段代码使用MinMaxScaler对训练数据进行缩放，然后将该缩放器应用于所有数组。这样做是为了防止数据泄漏。

### from sklearn.preprocessing import MinMaxScaler
### scaler = MinMaxScaler()
#使用scaler.fit_transform()函数对训练数据进行拟合和转换操作，然后对验证数据和测试数据使用scaler.transform()函数进行转换操作。
# x这是标准化
X_train_shifted = scaler.fit_transform(X_train_shifted)#x这个处理必须经过----------------,注意，transform会记忆上一个fit_transform的格式，所以全部改成fit_transform，否则报错

X_valid_shifted = scaler.fit_transform(X_valid_shifted)
X_test_shifted = scaler.fit_transform(X_test_shifted)

# x数组：转换为 LSTM 预期的 3D 数组
# x拆成（17500，1，35）,变成了一万七千组，一行为一组的数据
X_train_shifted = X_train_shifted.reshape(n_samples_train*n_time_steps, lag_steps, n_features_shifted)
X_valid_shifted = X_valid_shifted.reshape(n_samples_valid*n_time_steps, lag_steps, n_features_shifted)
X_test_shifted = X_test_shifted.reshape(n_samples_test*n_time_steps, lag_steps, n_features_shifted)

# y 数组：转换为常规 2D 数组：y_array（data_lines、n_labels）
# x拆成（17500，5）
y_train_shifted = y_train_shifted.reshape(n_samples_train*n_time_steps, n_labels)
y_valid_shifted = y_valid_shifted.reshape(n_samples_valid*n_time_steps, n_labels)
y_test_shifted = y_test_shifted.reshape(n_samples_test*n_time_steps, n_labels)
print("X_train_shifted形状",X_train_shifted.shape)
print("y_train_shifted形状",y_train_shifted.shape)
print("X_valid_shifted形状",X_valid_shifted.shape)
print("y_valid_shifted形状",y_valid_shifted.shape)
print("我看看x内容:",X_train_shifted[0][:])
print("我看看y内容:",y_train_shifted[0][:])
print("我看看预测的形状",X_valid_shifted.shape, y_valid_shifted.shape)
print("我看看预测的内容",X_valid_shifted[0][0][:], y_valid_shifted[0][:])
# print(X_train_shifted.shape, y_train_shifted.shape)
# print(X_valid_shifted.shape, y_valid_shifted.shape)
# print(X_test_shifted.shape, y_test_shifted.shape)
# print()

#lstm模型设置
input_shape0, input_shape1, input_shape2 = X_train_shifted.shape[0], X_train_shifted.shape[1], X_train_shifted.shape[2]
#print(input_shape0, input_shape1, input_shape2)
output_shape0, output_shape1 = y_train_shifted.shape[0], y_train_shifted.shape[1]
#print(output_shape0, output_shape1)

model = Sequential()
# Add a LSTM layer with 128 internal units.
model.add(LSTM(128, input_shape=(X_train_shifted.shape[1], X_train_shifted.shape[2])))
# Add a Dense layer with 5 units.
model.add(Dense(y_train_shifted.shape[1], activation='linear'))# x输出五个数，分别代表五个手指的弯曲角度
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
batch_size = 5000 # 这是每个独立测量集的持续时间 = 5000/(50 Hz) = 100 seconds, 每个t
epochs = 300

##打印语句来显示训练时间和训练结果
start_time = time.time()

history5_lstm = model.fit(X_train_shifted, y_train_shifted, batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(X_valid_shifted, y_valid_shifted), shuffle=False)

end_time = time.time()
total_time = end_time - start_time
print("Training time: ", total_time)

print(history5_lstm.history)  # 打印训练历史记录

#画图
def plot_history(history):
    # -----------------------------------------------------------
    # Retrieve results on training and validation data sets
    # for each training epoch
    # -----------------------------------------------------------

    mse = history.history['mean_squared_error']
    val_mse = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(mse) + 1)

    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # ------------------------------------------------
    # Plot training and validation mse per epoch
    # ------------------------------------------------
    ax1.plot(epochs, mse, label='Training mse')
    ax1.plot(epochs, val_mse, label='Validation mse')
    ax1.set_title('Loss = MSE')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mse')
    # ax1.set_ylim(0.5,1.2)
    ax1.legend()

    # ------------------------------------------------
    # Plot training and validation rmse per epoch
    # ------------------------------------------------
    ax2.plot(epochs, np.sqrt(mse), label='Training rmse')
    ax2.plot(epochs, np.sqrt(val_mse), label='Validation rmse')
    ax2.set_title('RMSE')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('rmse')
    # ax2.set_ylim(0.5,1.2)
    ax2.legend()

    fig.tight_layout()
    plt.show()

plot_history(history5_lstm)#画图调用



##评估模型在测试集上的性能    #模型 fit5 的最终结果，具有矢量自回归和 LSTM，典型误差约为 5 度（取决于手指和弯曲角度）
test_loss, test_mse = model.evaluate(X_test_shifted, y_test_shifted, batch_size=batch_size, verbose=2)
test_rmse = np.sqrt(test_mse)
y_test_shifted_pred = model.predict(X_test_shifted)
test_mae = mean_absolute_error(y_test_shifted, y_test_shifted_pred)
print('test_loss = test_mse, test_rmse, test_mae')
print("          %8.4f  %8.4f  %8.4f" % (test_mse, test_rmse, test_mae))
model.save('model.h5')
print("保存成功")

