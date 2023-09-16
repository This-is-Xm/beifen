import scipy.io as scio
data=scio.loadmat(r'C:\Users\xsz\Desktop\数据集\s1\S1_E1_A1.mat')
#读取csv，如果复现，路径记得要改
x = np.loadtxt(r"C:\Users\xsz\Desktop\cs171.csv",delimiter=",",usecols=[0,1,2,3,4,5])
y_train = np.loadtxt(r"C:\Users\xsz\Desktop\cs171y.csv",delimiter=",",usecols=[0])
