#以下为我个人的注解，仅供参考，另一个thesis_code_realistic_five_movement.py更像是此代码的原始版本（分类类数比此文件少）
from __future__ import print_function
from collections import deque
from threading import Lock, Thread
import matplotlib#进行数据处理的库
matplotlib.use("TkAgg")#让代码不显示图片
import matplotlib.pyplot as plt

import numpy as np#构造矩阵的库
#np.random.seed(1)
import tensorflow as tf#深度学习的库，keras同理
from tensorflow import keras
from keras import regularizers
from keras.models import load_model

from sklearn import preprocessing

import myo  #这个是myo自带的库
import time
import sys
import psutil#查看系统硬件状态的库
import os

#用于蓝牙连接
# Used for bluetooth signal sending
import serial


# This training set will contain 1000 samples of 8 sensor values
global training_set
global number_of_samples
global index_training_set, middle_training_set,thumb_training_set,verification_set
global data_array
number_of_samples = 1000
data_array=[]

# 8 Sensors in armband#创造8个1X1000全零矩阵
Sensor1 = np.zeros((1,number_of_samples))
Sensor2 = np.zeros((1,number_of_samples))
Sensor3 = np.zeros((1,number_of_samples))
Sensor4 = np.zeros((1,number_of_samples))
Sensor5 = np.zeros((1,number_of_samples))
Sensor6 = np.zeros((1,number_of_samples))
Sensor7 = np.zeros((1,number_of_samples))
Sensor8 = np.zeros((1,number_of_samples))

# 12 finger movements12种手势判断8X1000全零矩阵
index_open_training_set = np.zeros((8,number_of_samples))#食指
middle_open_training_set = np.zeros((8,number_of_samples))#中指
thumb_open_training_set = np.zeros((8,number_of_samples))#大拇指
ring_open_training_set = np.zeros((8,number_of_samples))#无名指
pinky_open_training_set = np.zeros((8,number_of_samples))#小拇指
two_open_training_set = np.zeros((8,number_of_samples))#两根手指头
three_open_training_set = np.zeros((8,number_of_samples))#三根手指头
four_open_training_set = np.zeros((8,number_of_samples))#四个手指头
five_open_training_set = np.zeros((8,number_of_samples))#五个手指头
all_fingers_closed_training_set = np.zeros((8,number_of_samples))#握拳
grasp_training_set = np.zeros((8,number_of_samples))#抓的手势
pick_training_set = np.zeros((8,number_of_samples))#挑选的手势

verification_set = np.zeros((8,number_of_samples))#检验设置
training_set = np.zeros((8,number_of_samples))#尝试设置（这俩不确定）

#给每个手势来一个标签
thumb_open_label = 0
index_open_label = 1
middle_open_label = 2
ring_open_label = 3
pinky_open_label = 4
two_open_label = 5
three_open_label = 6
four_open_label = 7
five_open_label = 8
all_fingers_closed_label = 9
grasp_label = 10
pick_label = 11
#输入主题名称
name = input("Enter name of Subject")
#检查是否exe(连接)正常运行
# Check if Myo Connect.exe process is running
def check_if_process_running():

    try:
        for proc in psutil.process_iter():
            if proc.name()=='Myo Connect.exe':
                return True
            
        return False
            
    except (psutil.NoSuchProcess,psutil.AccessDenied, psutil.ZombieProcess):
        print (PROCNAME, " not running")

#exe程序无法执行时，重启exe程序
# Restart myo connect.exe process if its not running
def restart_process():
    PROCNAME = "Myo Connect.exe"

    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == PROCNAME:
            proc.kill()
            # Wait a second
            time.sleep(1)

    while(check_if_process_running()==False):
        path = r'D:\Myo Connect\Myo Connect.exe'
        os.startfile(path)
        time.sleep(1)
        #while(check_if_process_running()==False):
        #    pass

    print("Process started")
    return True

# This class from Myo-python SDK listens to EMG signals from armband
class Listener(myo.DeviceListener):
    
    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def on_connected(self, event):
        print("Myo Connected")
        self.started = time.time()
        event.device.stream_emg(True)
        
    def get_emg_data(self):
        with self.lock:
            print("H")   # Ignore this

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.emg))
            
            if len(list(self.emg_data_queue))>=number_of_samples:
                data_array.append(list(self.emg_data_queue))
                self.emg_data_queue.clear()
                return False


#训练肌电图数据函数
# This method is responsible for training EMG data
def Train(conc_array):
    global training_set
    global index_open_training_set, middle_open_training_set, thumb_open_training_set, ring_open_training_set, pinky_open_training_set, verification_set
    global two_open_training_set, three_open_training_set, four_open_training_set,five_open_training_set,all_fingers_closed_training_set,grasp_training_set,pick_training_set
    global number_of_samples
    verification_set = np.zeros((8,number_of_samples))#验证集
    print (number_of_samples)
    
    labels = []
        
    print(conc_array,conc_array.shape)
    #使标签的迭代器在内循环中运行30次，在外循环中运行10次，总共运行300次，用于10个手指的移动
    # This division is to make the iterator for making labels run 30 times in inner loop and 10 times in outer loop running total 300 times for 10 finger movements
    samples = conc_array.shape[0]/12
    #开始制作训练集
    # Now we append all data in training label
    # We iterate to make 12 finger movement labels.
    #放入12个标签
    for i in range(0,12):
        for j in range(0,int(samples)):
            labels.append(i)
    #numpy数组转化（便于之后训练的）
    labels = np.asarray(labels)
    print(labels, len(labels),type(labels))
    print(conc_array.shape[0])
    #打乱数据,这是个seed
    permutation_function = np.random.permutation(conc_array.shape[0])

    total_samples = conc_array.shape[0]
    #这里是个形状转换，可能是之后不用换形状所以这里没用
    all_shuffled_data,all_shuffled_labels = np.zeros((total_samples,8)),np.zeros((total_samples,8))
     #对数据集和标签集打乱处理，数组[seed]格式
    all_shuffled_data,all_shuffled_labels = conc_array[permutation_function],labels[permutation_function]
    print(all_shuffled_data.shape)
    print(all_shuffled_labels.shape)
    #取80%数据当作训练集,这里保存训练数量的80%向下取整的数值（即10取8），注意这里是数值，不是数据集
    number_of_training_samples = np.int(np.floor(0.8*total_samples))
    #形状转换，没用上
    train_data = np.zeros((number_of_training_samples,8))
    train_labels = np.zeros((number_of_training_samples,8))
    print("TS ", number_of_training_samples, " S " , number_of_samples)
    number_of_validation_samples = np.int(total_samples-number_of_training_samples)
    #给数据换了一个变量名字，其它没变化,取80%分割
    train_data = all_shuffled_data[0:number_of_training_samples,:]
    train_labels = all_shuffled_labels[0:number_of_training_samples,]
    print("Length of train data is ", train_data.shape)
    validation_data = all_shuffled_data[number_of_training_samples:total_samples,:]
    validation_labels = all_shuffled_labels[number_of_training_samples:total_samples,]
    print("Length of validation data is ", validation_data.shape , " validation labels is " , validation_labels.shape)
    print(train_data,train_labels)        
    #终于开始模型建立
    model = keras.Sequential([
    # Input dimensions means input columns. Here we have 8 columns, one for each sensor
    #一个普普通通的全神经网络
    keras.layers.Dense(8, activation=tf.nn.relu,input_dim=8,kernel_regularizer=regularizers.l2(0.1)),
    #即使得其输出数据的均值接近0，其标准差接近1,能够加快收敛，效果比dropout厉害，新东西
    keras.layers.BatchNormalization(),
    keras.layers.Dense(12, activation=tf.nn.softmax)])
    #一个adam优化器，一个交叉熵损失函数，一个测试器用来看正确率
    adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    #模型调完了，开始学了，保存成h5文件，注意这里的路径，要改——————————————————————————————————————————————————————————————————————-———————————————————————————————————————————————————————————————

    history = model.fit(train_data, train_labels, epochs=300,validation_data=(validation_data,validation_labels),batch_size=16)
    model.save('D:\桌面\myo\Finger-Movement-Classification-via-Machine-Learning-using-EMG-Armband-for-3D-Printed-Robotic-Hand-master/'+name+'_realistic_model.h5')
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
    #这里的解释是初始化，但我暂时没能理解这步的意义，200/50=4=average，创造了一个四行八列的全零数组，下面也是/50
    averages = number_of_samples/50
    # Initializing array for verification_averages
    verification_averages = np.zeros((int(averages),8))
    
    
    while True:
        while True:
            try:
                input("Hold a finger movement and press enter to get its classification")
                #这里用的是myo自带的库，暂时无法解析
                hub = myo.Hub()        
                number_of_samples=200
                listener = Listener(number_of_samples)
                hub.run(listener.on_event,20000)

                #接收样本数量，转为一个1000行8列的numpy列表
                # Here we send the received number of samples making them a list of 1000 rows 8 columns
                verification_set = np.array((data_array[0]))
                data_array.clear()
                break
            except:
                #等待，直到上面那句正确执行
                while(restart_process()!=True):
                    pass
                # Wait for 3 seconds until Myo Connect.exe starts
                time.sleep(3)
        #对验证集求绝对值，可能是因为训练数据不允许负数存在
        verification_set = np.absolute(verification_set)

        div = 50
        #与上面的/50对应，这里可能是反复调参得出的结果，将数据分成了五份，然后全部展平，用法与flatten()类似
        # We add one because iterator below starts from 1
        batches = int(number_of_samples/div) + 1
        for i in range(1,batches):
            verification_averages[i-1,:] = np.mean(verification_set[(i-1)*div:i*div,:],axis=0)

        verification_data = verification_averages
        print("Verification matrix shape is " , verification_data.shape)
        #从这里开始预测结果，下面标签对应的我就不翻译了，英文也能看懂
        predictions = model.predict(verification_data,batch_size=16)
        predicted_value = np.argmax(predictions[0])
        print(predictions[0])
        print(predicted_value)
        if predicted_value == 0:
            print("Thumb open")
        elif predicted_value == 1:
            print("Index finger open")
        elif predicted_value == 2:
            print("Middle finger open")
        elif predicted_value == 3:
            print("Ring finger open")
        elif predicted_value == 4:
            print("Pinky finger open")
        elif predicted_value == 5:
            print("Two fingers open")
        elif predicted_value == 6:
            print("Three fingers open")
        elif predicted_value == 7:
            print("Four fingers open")
        elif predicted_value == 8:
            print("Five fingers open")
        elif predicted_value == 9:
            print("All fingers closed")
        elif predicted_value == 10:
            print("Grasp movement")
        elif predicted_value == 11:
            print("Pick movement")
        else:
            print("pass")
            pass
        # #通过蓝牙将预测值发送到arduino，驱动打开对应的手指
        # #### Here i send the predicted value to Arduino via Bluetooth so that it can open appropriate fingers ####
        # #这里持续循环是因为蓝牙有时候会在打开COM的时候会抛出异常，所以需要一直尝试直到数据发送和接收数据
        # # While 1 is used because sometimes bluetooth port throws exception in opening the COM Port
        # # So i keep trying until the data is sent and confirmation received.
        #
        # #这个就到我的知识盲区了，是硬件控制代码，只能说是知道在接收处理数据，具体不清楚，靠你们了
        # while(1):
        #     try:
        #         # Bluetooth at COM6
        #         print("111")
        #         serialPort = serial.Serial(port="COM6",baudrate=9600,bytesize=8,timeout=20,stopbits=serial.STOPBITS_ONE)
        #         print("222")
        #         value_to_bluetooth = str(predicted_value).encode()
        #         if predicted_value == 10:
        #             value_to_bluetooth = 'a'.encode()
        #         if predicted_value == 11:
        #             value_to_bluetooth = 'b'.encode()
        #         print(value_to_bluetooth)
        #         #调到了这里，不卡死，不报错，是持续写入（注意这里的write是自定义函数），我认为这里可能是得创建一个传入端口，端口借一个接收器来接收数据
        #         serialPort.write(str(value_to_bluetooth))
        #         print("444")
        #         # time.sleep(1)
        #         if serialPort.in_waiting>0:
        #             serialString = serialPort.readline()
        #             print(serialString)
        #             #人话：接收数据要么全对，要么全扔掉
        #             # If we receive what we sent from Arduino bluetooth then all OK else bad value
        #             if serialString == value_to_bluetooth:
        #                 print("Received")
        #             else:
        #                 print("Bad value")
        #         serialPort.close()
        #         break
        #     except serial.SerialException as e:
        #         #There is no new data from serial port
        #         print (str(e))
        #     except TypeError as e:
        #         print (str(e))
        #         ser.port.close()
            

def main():
    #一万个全零矩阵（不是），12个手势对照组，检验和尝试对照组（同样的，这两个不确定）
    unrecognized_training_set = np.zeros((8,number_of_samples))
    index_open_training_set = np.zeros((8,number_of_samples))
    middle_open_training_set = np.zeros((8,number_of_samples))
    thumb_open_training_set = np.zeros((8,number_of_samples))
    ring_open_training_set = np.zeros((8,number_of_samples))
    pinky_open_training_set = np.zeros((8,number_of_samples))
    two_open_training_set = np.zeros((8,number_of_samples))
    three_open_training_set = np.zeros((8,number_of_samples))
    four_open_training_set = np.zeros((8,number_of_samples))
    five_open_training_set = np.zeros((8,number_of_samples))
    all_fingers_closed_training_set = np.zeros((8,number_of_samples))
    grasp_training_set = np.zeros((8,number_of_samples))
    pick_training_set = np.zeros((8,number_of_samples))
    
    verification_set = np.zeros((8,number_of_samples))
    
    
    training_set = np.zeros((8,number_of_samples))
    #翻译：为了防止myo connect.exe出故障（却被识别正常运行），重启它。
    # This function kills Myo Connect.exe and restarts it to make sure it is running
    # Because sometimes the application does not run even when Myo Connect process is running
    # So i think its a good idea to just kill if its not running and restart it

    while(restart_process()!=True):
        pass
    # Wait for 3 seconds until Myo Connect.exe starts
    time.sleep(3)
    #初始化SDK
    # Initialize the SDK of Myo Armband
    myo.init('C:\Windows\System32\\myo64.dll')
    hub = myo.Hub()
    listener = Listener(number_of_samples)
    #足足八个传感器
    legend = ['Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7','Sensor 8']
    #大拇指打开数据
    ################## HERE WE GET TRAINING DATA FOR THUMB FINGER OPEN ########
    while True:
        try:
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            input("Open THUMB ")    
            hub.run(listener.on_event,20000)
            thumb_open_training_set = np.array((data_array[0]))
            print(thumb_open_training_set.shape)
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
        
    # Here we send the received number of samples making them a list of 1000 rows 8 columns just how we need to feed to tensorflow
    #食指打开数据
    ################## HERE WE GET TRAINING DATA FOR INDEX FINGER OPEN ########
    while True:
        try:
            input("Open index finger")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            # Here we send the received number of samples making them a list of 1000 rows 8 columns 
            index_open_training_set = np.array((data_array[0]))            
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #中指打开数据
    ################## HERE WE GET TRAINING DATA FOR MIDDLE FINGER OPEN #################
    while True:
        try:
            input("Open MIDDLE finger")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            middle_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)

    # Here we send the received number of samples making them a list of 1000 rows 8 columns
    #无名指打开数据
    ################## HERE WE GET TRAINING DATA FOR RING FINGER OPEN ##########
    while True:
        try:
            input("Open Ring finger")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            ring_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #小拇指打开数据
    ################### HERE WE GET TRAINING DATA FOR PINKY FINGER OPEN ####################
    while True:
        try:
            input("Open Pinky finger")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            pinky_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #两根手指打开数据
    ################### HERE WE GET TRAINING DATA FOR TWO FINGER OPEN ####################
    while True:
        try:
            
            input("Open Two fingers")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            two_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #三根手指打开数据
    ################### HERE WE GET TRAINING DATA FOR THREE FINGER OPEN ####################
    while True:
        try:
            input("Open Three fingers")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            three_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #这里注释打错了，是四根手指打开数据
    ################### HERE WE GET TRAINING DATA FOR THREE FINGER OPEN ####################
    while True:
        try:            
            input("Open Four fingers")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            four_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #五根手指打开数据
    ################### HERE WE GET TRAINING DATA FOR FIVE FINGER OPEN ####################
    while True:
        try:
            input("Open Five fingers")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            five_open_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #五个手指并拢，握拳
    ################### HERE WE GET TRAINING DATA FOR ALL FINGERS CLOSED ####################
    while True:
        try:
            input("Make all fingers closed")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            all_fingers_closed_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #五个手指并拢，握拳
    ################### HERE WE GET TRAINING DATA FOR GRASP MOVEMENT ####################
    while True:
        try:
            input("Make Grasp movement")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            grasp_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #抓取姿势
    ################### HERE WE GET TRAINING DATA FOR PICK MOVEMENT ####################
    while True:
        try:
            input("Make Pick movement")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            hub.run(listener.on_event,20000)
            pick_training_set = np.array((data_array[0]))
            data_array.clear()
            break
        except:
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)
    #对所有合拢手指数据进行绝对值处理，下面是打开的，处理方法一样
    # Absolute of finger open data
    thumb_open_training_set = np.absolute(thumb_open_training_set)
    index_open_training_set = np.absolute(index_open_training_set)
    middle_open_training_set = np.absolute(middle_open_training_set)
    ring_open_training_set = np.absolute(ring_open_training_set)
    pinky_open_training_set = np.absolute(pinky_open_training_set)
    # Absolute of finger close data
    two_open_training_set = np.absolute(two_open_training_set)
    three_open_training_set = np.absolute(three_open_training_set)
    four_open_training_set = np.absolute(four_open_training_set)
    five_open_training_set = np.absolute(five_open_training_set)
    all_fingers_closed_training_set = np.absolute(all_fingers_closed_training_set)
    grasp_training_set = np.absolute(grasp_training_set)
    pick_training_set = np.absolute(pick_training_set)
    #都是对照组，跟上面一样，是全零矩阵
    div = 50
    averages = int(number_of_samples/div)
    thumb_open_averages = np.zeros((int(averages),8))
    index_open_averages = np.zeros((int(averages),8))
    middle_open_averages = np.zeros((int(averages),8))
    ring_open_averages = np.zeros((int(averages),8))
    pinky_open_averages = np.zeros((int(averages),8))
    two_open_averages = np.zeros((int(averages),8))
    three_open_averages = np.zeros((int(averages),8))
    four_open_averages = np.zeros((int(averages),8))
    five_open_averages = np.zeros((int(averages),8))
    all_fingers_closed_averages = np.zeros((int(averages),8))
    grasp_averages = np.zeros((int(averages),8))
    pick_averages = np.zeros((int(averages),8))
    #翻译：计算所有手指张开数据的平均值，并将其存储为n/50个样本，因为50批样本等于n/50个平均值（？）。我认为与上面相同是对数据的展平处理，这便于之后的数据连接
    # Here we are calculating the mean values of all finger open data set and storing them as n/50 samples because 50 batches of n samples is equal to n/50 averages
    for i in range(1,averages+1):
        thumb_open_averages[i-1,:] = np.mean(thumb_open_training_set[(i-1)*div:i*div,:],axis=0)
        index_open_averages[i-1,:] = np.mean(index_open_training_set[(i-1)*div:i*div,:],axis=0)
        middle_open_averages[i-1,:] = np.mean(middle_open_training_set[(i-1)*div:i*div,:],axis=0)
        ring_open_averages[i-1,:] = np.mean(ring_open_training_set[(i-1)*div:i*div,:],axis=0)
        pinky_open_averages[i-1,:] = np.mean(pinky_open_training_set[(i-1)*div:i*div,:],axis=0)

        two_open_averages[i-1,:] = np.mean(two_open_training_set[(i-1)*div:i*div,:],axis=0)
        three_open_averages[i-1,:] = np.mean(three_open_training_set[(i-1)*div:i*div,:],axis=0)
        four_open_averages[i-1,:] = np.mean(four_open_training_set[(i-1)*div:i*div,:],axis=0)
        five_open_averages[i-1,:] = np.mean(five_open_training_set[(i-1)*div:i*div,:],axis=0)
        all_fingers_closed_averages[i-1,:] = np.mean(all_fingers_closed_training_set[(i-1)*div:i*div,:],axis=0)
        grasp_averages[i-1,:] = np.mean(grasp_training_set[(i-1)*div:i*div,:],axis=0)
        pick_averages[i-1,:] = np.mean(pick_training_set[(i-1)*div:i*div,:],axis=0)                
        
     
    # Here we stack all the data row wise
    #行方式连接所有数据形成原始数据集
    conc_array = np.concatenate([thumb_open_averages,index_open_averages,middle_open_averages,ring_open_averages,pinky_open_averages,two_open_averages,three_open_averages,four_open_averages,five_open_averages,all_fingers_closed_averages,grasp_averages,pick_averages],axis=0)
    # print conc_array(.shape)
    np.savetxt(r'C:\Users\xsz\Desktop\\'+name+'.txt', conc_array, fmt='%i')
    #训练从这里开始
    # In this method the EMG data gets trained and verified
    Train(conc_array)

if __name__ == '__main__':
    main()
