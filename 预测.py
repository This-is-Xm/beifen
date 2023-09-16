import numpy as np
from keras.models import load_model
from numpy import argmax
from sklearn.preprocessing import MinMaxScaler

model_xsz = load_model(r"C:\Users\xsz\Desktop\pytorch学习\假肢3.0\3.0model171.h5")

x = np.loadtxt(r"C:\Users\xsz\Desktop\cs191.csv",delimiter=",",usecols=[0,1,2,3,4,5])
y = np.loadtxt(r"C:\Users\xsz\Desktop\总集y.csv",delimiter=",",usecols=[0])
np.random.seed(118)
np.random.shuffle(x)
np.random.seed(118)
np.random.shuffle(y)
# mm = MinMaxScaler(feature_range=(0,1))
# x=mm.fit_transform(x)
right=0
rows=[]
for num in range(6):
    for i in range(len(x)//6*num,len(x)//6*(num+1)):
        predict = model_xsz.predict(x[i].reshape(1, -1))
        if argmax(predict)==int(y[i]):
            right+=1
    print(right,len(x)//6)
    print("标签:",num,"正确率:",right*100//(len(x)//6),"%")
    right=0

