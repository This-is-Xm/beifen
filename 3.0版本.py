import numpy as np
import csv
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import regularizers
import tensorflow as tf
import pandas as pd
from scipy.signal import savgol_filter
from keras.callbacks import EarlyStopping
labels = np.loadtxt(r"C:\Users\xsz\Desktop\cs129y.csv",delimiter=",",usecols=[0])
labels_pd=pd.DataFrame(labels)
num=min(labels_pd[0].value_counts())#取最小行数，3780
#print(labels_pd[0].value_counts()[3])#取特定标签的行数
#print(labels_pd[0].value_counts())

# #保存一下数据
# np.savetxt(r'C:\Users\xsz\Desktop\y.csv',y, delimiter=",")

with open(r'C:\Users\xsz\Desktop\cs129.csv','r',encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

#print(len(rows),"rows:",len(rows[1])) #5364 rows: 50
row1=rows[0]
row2=rows[1]
row3=rows[2]
row4=rows[3]
row5=rows[4]
row6=rows[5]
y=list(labels[0:6])
print(labels[1])
start=6
end=num
for x in range(11):
    for i in range(start,end):
        if i % 6 == 0:
            row1.extend(rows[i])
        elif i % 6 == 1:
            row2.extend(rows[i])
        elif i % 6 == 2:
            row3.extend(rows[i])
        elif i % 6 == 3:
            row4.extend(rows[i])
        elif i % 6 == 4:
            row5.extend(rows[i])
        elif i % 6 == 5:
            row6.extend(rows[i])
        else:
            print("啊？")
        y.append(labels[i])
    print(i)
    if x==0:
        start=0
    start+=labels_pd[0].value_counts()[x]
    end=start+num
row1=np.array(row1)
row2=np.array(row2)
row3=np.array(row3)
row4=np.array(row4)
row5=np.array(row5)
row6=np.array(row6)
row=np.vstack((row1,row2,row3,row4,row5,row6))
#平滑处理
#row=savgol_filter(row, 5, 3, mode= 'nearest')

row=np.array(row,dtype = int)
y=np.array(y,dtype = int)
np.savetxt(r'C:\Users\xsz\Desktop\x.csv',row.T, delimiter=",", fmt='%s')
np.savetxt(r'C:\Users\xsz\Desktop\y.csv',y, delimiter=",", fmt='%s')

div=50
averages=(row.T.shape[0]/div)
average_numpy = np.zeros((int(averages),6))
for i in range(1,int(averages)+1):
    average_numpy[i - 1, :] = np.mean(row.T[(i - 1) * div:i * div, :], axis=0)
# print(average_numpy.shape)
# print(average_numpy[0])
labels=[]
samples = average_numpy.shape[0]/11
for i in range(0,11):
    for j in range(0,int(samples)):
        labels.append(i)
#numpy数组转化（便于之后训练的）
labels = np.asarray(labels)

# permutation_function = np.random.permutation(average_numpy.shape[0])
# x_train,y_train= average_numpy[permutation_function],labels[permutation_function]

# 终于开始模型建立
model = keras.Sequential([
#一个普普通通的全神经网络
keras.layers.Dense(6, activation=tf.nn.relu,input_dim=6,kernel_regularizer=regularizers.l2(0.01)),
#即使得其输出数据的均值接近0，其标准差接近1,能够加快收敛，效果比dropout厉害，新东西
keras.layers.Dense(11,activation='sigmoid'),
keras.layers.BatchNormalization(),
keras.layers.Dense(11, activation=tf.nn.softmax)])
#一个adam优化器，一个交叉熵损失函数，一个测试器用来看正确率
adam_optimizer = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

#打乱
np.random.seed(116)
np.random.shuffle(average_numpy)
np.random.seed(116)
np.random.shuffle(labels)
print(average_numpy.shape)
    #模型调完了，开始学了，保存成h5文件，注意这里的路径，要改——————————————————————————————————————————————————————————————————————-———————————————————————————————————————————————————————————————
    #做个80%的分割
# callback=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto', restore_best_weights=True)#这个earlyStopping百分百优化，所以添上去不改了
history = model.fit(average_numpy, labels, epochs=200,validation_split=0.8,batch_size=64)
# model.save('D:\桌面\myo\Finger-Movement-Classification-via-Machine-Learning-using-EMG-Armband-for-3D-Printed-Robotic-Hand-master/'+name+'_realistic_model.h5')
# model.load_model('my_model.h5')
#画个图直观一点，看一下收敛情况
# Here we display the training and test loss for model
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