import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import regularizers
import matplotlib.pyplot as plt

#读取csv，如果复现，路径记得要改
x = np.loadtxt(r"C:\Users\xsz\Desktop\cs191.csv",delimiter=",",usecols=[0,1,2,3,4,5])
y_train = np.loadtxt(r"C:\Users\xsz\Desktop\总集y.csv",delimiter=",",usecols=[0])
#平滑处理，目前训练不怎么影响数据，注释
#x=savgol_filter(x, 5, 3, mode= 'nearest')
#归一化，将数据归类进0，1之间，显著加快拟合速度
# mm = MinMaxScaler(feature_range=(0,1))
# x=mm.fit_transform(x)
#赋值
x_train=x
# print(x.shape)
# print(y_train.shape)

#注：如果标签不为0的话，请以零开始，按顺序作为标签，否则会训练不出结果
# y_train=[]
# list=[0,1,2]
# for z in list:
#     for i in range(600):
#         y_train.append(z)
# y_train=np.array(y_train)
# print(x_train.shape,y_train[0:10])

#打乱
np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
#构建网络
model = keras.Sequential([
#输入层，6通道，input_dim与input_shape效果类似，input_dim表示输入列数，kernel_regularizer正则化，在此用于降低过拟合
keras.layers.Dense(6, activation=tf.nn.relu,input_dim=6,kernel_regularizer=regularizers.l2(0.1)),
#即使得其输出数据的均值接近0，其标准差接近1,能够加快收敛
keras.layers.BatchNormalization(),
#激活sigmoid
keras.layers.Dense(6,activation='sigmoid'),
keras.layers.BatchNormalization(),
#输出层，分为6类，激活sofmax
keras.layers.Dense(6, activation=tf.nn.softmax)])

#自定义adam优化器，效果没有adam好，注释
#adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#一个adam优化器，loss交叉熵损失函数，mertrics测试器用来看正确率
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
#earlyStopping提早停止，在防止过拟合
#callback=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', restore_best_weights=True)
#训练，目前最佳函数'adam'最佳batch_size=128 ，正确率95%
history = model.fit(x_train, y_train, epochs=700,validation_split=0.9,batch_size=128)
#画图
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#保存数据，如果要复现，路径记得要改
model.save(r'C:\Users\xsz\Desktop\pytorch学习\假肢3.0\3.0model171.h5')