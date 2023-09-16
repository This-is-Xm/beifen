from keras import Sequential
from keras.layers import Embedding, LSTM
from keras.layers import Input, Dense, Activation, Dropout
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt

# #改成卢紫腾数据的处理方式了，先运行下方注释的，然后再运行上方注释的
features = np.loadtxt(r"F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\训练集x.csv",delimiter=",",usecols=[0,1,2])    # 调取第123列
labels = np.loadtxt(r"F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\训练集y.csv",delimiter=",",usecols=[0])          # 调取第1列
x_shifted=np.full((len(features)-100,len(features[0])+6), np.NaN)
a_all =[]
s_all =[]
a = 0
s = 0
x=features
y=[]
for i in range(1,len(labels)):
    y.append(labels[i])

#预处理
lenx=len(x)
for time in range(100,len(x)):
    print("正在运行",time,lenx)
    for m in range(3):
        for i in range(10):#取前十个样本
            a +=x[time-100+i][m]
        a=a/10
        a_all.append(a)

        for i in range(100):#取前一百个样本
            s +=x[time-100+i][m]
        s=s/100
        s_all.append(s)
    x_shifted[time-100]=np.hstack((x[time], a_all,s_all))
    a_all=[]
    s_all=[]


print(x_shifted.shape)
#保存数据
np.savetxt(r'F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\符合输入格式的数据xx.csv',x_shifted, delimiter=",")
np.savetxt(r'F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\符合输入格式的数据yy.csv',y, delimiter=",")



# 这里是分割线
import csv
# # y数据处理
# labels = np.loadtxt(r"F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\data6--plus-by-Sun-Cheng - 提取y.csv",delimiter=",",usecols=[0])
# y=[]
# for i in range(len(labels)):
#     if i%6==0:
#         num=((labels[i]+labels[i+1]+labels[i+2])/3+(labels[i+3]+labels[i+4]+labels[i+5])/3)/2#把50：3先求平均值变成50：1再求平均值变成100：1
#         y.append(num)
# num=int(len(y)*Decimal('0.9'))  # Python乘除小数会以近似值来表示，原来的代码会导致一个bug，比如50*0.9不是45而是44.9999996，而int（）转换会直接去掉小数部分，就导致了计算错误，decimal是强制让小数四舍五入，就可以避免这个bug
# print("num",num)
# y1=y[:num]
# y2=y[num:]
# #保存一下数据
# np.savetxt(r'F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\训练集y.csv',y1, delimiter=",")
# np.savetxt(r'F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\测试集y.csv',y2, delimiter=",")
#
# with open(r'F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\data6--plus-by-Sun-Cheng- 去掉y.csv','r',encoding="utf-8") as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]
# #太多了，分割一下
# num=int(len(rows)*Decimal('0.9'))    # 这个decimal是用来限制0.7是0.7而不是0.69999其他什么的
# row_x=rows[:num]       # 分离出训练集用的X
# row_y=rows[num:]       # 分离出测试集用的X
# row1=row_x[0]
# row2=row_x[1]
# row3=row_x[2]
# for i in range(3,len(row_x)):
#     if i%3==0:
#         row1.extend(row_x[i])
#     elif i%3==1:
#         row2.extend(row_x[i])
#     elif i%3==2:
#         row3.extend(row_x[i])
#         print(i)
#     else:
#         print("啊？")
# row1=np.array(row1)
# row2=np.array(row2)
# row3=np.array(row3)
# row=np.vstack((row1,row2,row3))
# #保存数据
# np.savetxt(r'F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\训练集x.csv',row.T, delimiter=",", fmt='%s')       # 注意转置操作在这里row.T
#                                                                                      # 最后只使用了训练集x，测试集x并没有使用 validation_split=0.7)
# row1=row_y[0]
# row2=row_y[1]
# row3=row_y[2]
# for i in range(3,len(row_y)):
#     if i%3==0:
#         row1.extend(row_y[i])
#     elif i%3==1:
#         row2.extend(row_y[i])
#     elif i%3==2:
#         row3.extend(row_y[i])
#         print(i)
#     else:
#         print("啊？")
# row1=np.array(row1)
# row2=np.array(row2)
# row3=np.array(row3)
# row=np.vstack((row1,row2,row3))
# #保存数据
# np.savetxt(r"F:\AI\肌肉电假肢-3代\徐顺涨初期3通道识别1手指弯曲度\6通道肌肉电数据及代码\保存的新数据\测试集x.csv",row.T, delimiter=",", fmt='%s')    # 注意转置操作在这里row.T
#                                                                              # 最后只使用了训练集x，测试集x并没有使用 validation_split=0.7)

#  生成完训练集x.csv 训练集y.csv,测试集x.csv 测试集y.csv  后要自行修正数据x的个数要能被100整除而不是50，同时删除y