from keras import Sequential
from keras.layers import Embedding, LSTM
from keras.layers import Input, Dense, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import csv

# #改成卢紫腾数据的处理方式了，先运行分割线下方注释的，然后再运行分割线上方注释的
# features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\训练集x.csv",delimiter=",",usecols=[0,1,2])
# labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\训练集y.csv",delimiter=",",usecols=[0])
# x_shifted=np.full((len(features)-100,len(features[0])+6), np.NaN)
# a_all =[]
# s_all =[]
# a = 0
# s = 0
# x=features
# y=[]
# for i in range(1,len(labels)):
# #     if i%6==0:
# #         num=((labels[i]+labels[i+1]+labels[i+2])/3+(labels[i+3]+labels[i+4]+labels[i+5])/3)/2#把50：3先求平均值变成50：1再求平均值变成100：1
#           y.append(labels[i])
#
# #预处理
# for time in range(100,len(x)):
#     for m in range(3):
#         for i in range(10):#取前十个样本
#             a +=x[time-10+i][m]
#         a=a/10
#         a_all.append(a)
#
#         for i in range(100):#取前一百个样本
#             s +=x[time-100+i][m]
#         s=s/100
#         s_all.append(s)
#     x_shifted[time-100]=np.hstack((x[time], a_all,s_all))
#     a_all=[]
#     s_all=[]
#
#
# print(x_shifted.shape)
##保存数据
# np.savetxt(r'C:\Users\xsz\Desktop\符合输入格式的数据xx.csv',x_shifted, delimiter=",")
# np.savetxt(r'C:\Users\xsz\Desktop\符合输入格式的数据yy.csv',y, delimiter=",")


# #这里是分割线
#y数据处理
labels = np.loadtxt(r"C:\Users\xsz\Desktop\总集y.csv",delimiter=",",usecols=[0])
y=[]
print(len(labels))
for i in range(len(labels)):
    if i%6==5:
        num=(labels[i-5]+labels[i-4]+labels[i-3]+labels[i-2]+labels[i-1]+labels[i])/6#把50：3先求平均值变成50：1再求平均值变成100：1
        y.append(num)
num=int(len(y)*Decimal('0.9'))
#print("num",num)
y1=y[:num]
y2=y[num:]
#保存一下数据
np.savetxt(r'C:\Users\xsz\Desktop\训练集y.csv',y1, delimiter=",")
np.savetxt(r'C:\Users\xsz\Desktop\测试集y.csv',y2, delimiter=",")

with open(r'C:\Users\xsz\Desktop\cs171.csv','r',encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
#太多了，分割一下
num=int(len(rows)*Decimal('0.9'))
row_x=rows[:num]
row_y=rows[num:]
row1=row_x[0]
row2=row_x[1]
row3=row_x[2]
for i in range(3,len(row_x)):
    if i%3==0:
        row1.extend(row_x[i])
    elif i%3==1:
        row2.extend(row_x[i])
    elif i%3==2:
        row3.extend(row_x[i])
        print(i)
    else:
        print("啊？")
row1=np.array(row1)
row2=np.array(row2)
row3=np.array(row3)
row=np.vstack((row1,row2,row3))
#保存数据
np.savetxt(r'C:\Users\xsz\Desktop\训练集x.csv',row.T, delimiter=",", fmt='%s')

row1=row_y[0]
row2=row_y[1]
row3=row_y[2]
for i in range(3,len(row_y)):
    if i%3==0:
        row1.extend(row_y[i])
    elif i%3==1:
        row2.extend(row_y[i])
    elif i%3==2:
        row3.extend(row_y[i])
        #print(i)
    else:
        print("啊？")
row1=np.array(row1)
row2=np.array(row2)
row3=np.array(row3)
row=np.vstack((row1,row2,row3))
#保存数据
np.savetxt(r'C:\Users\xsz\Desktop\测试集x.csv',row.T, delimiter=",", fmt='%s')