import numpy as np
from scipy import signal

# 滤波函数（可根据需要自定义）
def filter_signal(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.lfilter(b, a, signal)
    return filtered_signal

# 时域特征提取函数（可根据需要自定义）
def extract_time_domain_features(signal):
    mean_amplitude = np.mean(signal)
    std_amplitude = np.std(signal)
    rms = np.sqrt(np.mean(np.square(signal)))
    # 这里可以添加更多时域特征

    return mean_amplitude, std_amplitude, rms

# 频域特征提取函数（可根据需要自定义）
def extract_frequency_domain_features(signal, fs):
    freq_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fs)

    # 将频谱信号限制在感兴趣的频率范围内
    lowcut_freq = 10  # 设置低频截断阈值
    highcut_freq = 200  # 设置高频截断阈值
    freq_signal[(np.abs(freqs) < lowcut_freq)] = 0
    freq_signal[(np.abs(freqs) > highcut_freq)] = 0

    power_spectrum = np.abs(freq_signal) ** 2
    max_power = np.max(power_spectrum)
    mean_power = np.mean(power_spectrum)
    median_power = np.median(power_spectrum)
    # 这里可以添加更多频域特征

    return max_power, mean_power, median_power

# 测试数据
sEMG_signal = np.random.rand(1000)  # 示例：随机生成1000个样本的sEMG信号
sampling_frequency = 1000  # 采样频率，单位Hz
print()
# 滤波处理
filtered_signal = filter_signal(sEMG_signal, lowcut=20, highcut=500, fs=sampling_frequency)

# 提取时域特征
mean_amp, std_amp, rms = extract_time_domain_features(filtered_signal)

# 提取频域特征
max_power, mean_power, median_power = extract_frequency_domain_features(filtered_signal, sampling_frequency)

# 输出结果
print("时域特征:")
print("平均振幅:", mean_amp)
print("振幅标准差:", std_amp)
print("均方根值:", rms)

print("\n频域特征:")
print("最大功率:", max_power)
print("平均功率:", mean_power)
print("中位数功率:", median_power)