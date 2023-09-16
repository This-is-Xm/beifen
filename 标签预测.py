import numpy as np
from keras import regularizers
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf

features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据xx.csv",delimiter=",",usecols=[3,4,5,6,7,8])
labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\符合输入格式的数据yy.csv",delimiter=",",usecols=[0])
x_train=features.reshape(len(features),6)


# for i in range(len(x_train)):
#     mm = MinMaxScaler(feature_range=(0,10))
#     x_train[i]=mm.fit_transform(x_train[i])
for i in range(len(labels)):
    if labels[i]>=0 and labels[i]<45:
        labels[i] = 0
    elif labels[i]>=45 and labels[i]<75:
        labels[i] = 1
    elif labels[i] >= 75 and labels[i] < 105:
        labels[i] = 2
    elif labels[i]>=105 and labels[i]<135:
        labels[i] = 3
    else:
        labels[i] = 4

y_train=[]
for i in range(len(labels)):
    for x in range(100):
        y_train.append(labels[i])

y_train=np.array(y_train)
print(y_train.shape)
# #打乱
# np.random.seed(116)
# np.random.shuffle(x_train)
# np.random.seed(116)
# np.random.shuffle(y_train)
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
keras.layers.Dense(5, activation=tf.nn.softmax)])

#自定义adam优化器，效果没有adam好，注释
#adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#一个adam优化器，loss交叉熵损失函数，mertrics测试器用来看正确率
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
#earlyStopping提早停止，在防止过拟合
#callback=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', restore_best_weights=True)
#训练，目前最佳函数'adam'最佳batch_size=128 ，正确率95%
history = model.fit(x_train, y_train, epochs=200,validation_split=0.7,batch_size=256)

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