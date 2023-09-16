import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# features = np.loadtxt(r"C:\Users\xsz\Desktop\保存\训练集x.csv",delimiter=",",usecols=[0,1,2])
# #labels = np.loadtxt(r"C:\Users\xsz\Desktop\保存\训练集y.csv",delimiter=",",usecols=[0])
# y_smooth = scipy.signal.savgol_filter(features[:,0],99,9)
# x=[]
# for i in range(len(y_smooth)):
#     x.append(i)
# plt.plot(x,y_smooth)
# plt.plot(x,features[:,0])
# plt.show()

import h5py

#HDF5的读取：
file = h5py.File('DB1_S1_image.h5','r')   #打开h5文件
imageData   = file['imageData'][:]
imageLabel  = file['imageLabel'][:]
file.close()
print(imageData.shape)
print(imageLabel.shape)