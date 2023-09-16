import serial
import numpy as np
import csv

# set serial port initialized parameters
com = serial.Serial(
    port='COM5',
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
batch = 80
x_train = []
x_text = []
a1 = []
a2 = []
a = 0  # x_train(batch,50,3);的50
b = 0  # 同上3
c = 0
# received data and print in hex string form
while True:
    rxd = com.read(1)
    if a < 50:
        for i in rxd:
            a1.append(i)
        a += 1
    elif a == 50:
        a2.append(a1)
        a1=[]
        a += 1
        for i in rxd:
            if i != 221:
                print("不对")
    elif a==52:
        a = 0
        b += 1
        for i in rxd:
            x_text.append(i)
    else:
        a += 1
        for i in rxd:
            if i != 0:
                print("不对")
    if b == 3:
        x_train.append(a2)
        a2 = []
        c += 1
        print(c)
        b=0
    if c == batch:
        break
arr = np.array(x_train)
x_text = np.array(x_text)
x_train=np.zeros((batch,50,3),np.int16)
for i in range(batch):
    x_train[i] = np.transpose(arr[i])
B = np.reshape(x_train,(-1,3))
np.savetxt(r"C:\Users\xsz\Desktop\pytorch学习\假肢3.0\x1.csv", B, delimiter=",")
np.savetxt(r"C:\Users\xsz\Desktop\pytorch学习\假肢3.0\y1.csv", x_text, delimiter=",")