import numpy as np
import matplotlib.pyplot as plt

fs_overall = 10e9       # TI-ADC 總取樣率 (10 GHz)
Ts = 1 / fs_overall     # 總取樣週期 (100 ps)

N_samples = 1024        # 交錯後總取樣點數
N_fft = 2048            # 用於做 FFT 的長度

# (1) 調整輸入信號頻率為「對齊 FFT bin」的值
# bin_spacing = fs_overall / N_fft ≈ 4.88 MHz
# bin = 205 => 205 * 4.88 MHz ≈ 1 GHz
bin_index = 205
f_signal = bin_index * (fs_overall / N_fft)

A_signal = 0.5          # 輸入信號幅值 (0.5 V)

# 設定兩個通道的失配參數
# 時間偏移 (秒)
dt1 = 5e-12             # 通道 1 延遲 +5 ps
dt2 = -5e-12            # 通道 2 延遲 -5 ps
# 偏置失配 (伏特)
offset1 = 0.01          # 通道 1 偏置 +10 mV
offset2 = -0.01         # 通道 2 偏置 -10 mV
# 頻寬失配：使用一階低通濾波器模擬
# 通道 1 截止頻率: 3 GHz，通道 2 截止頻率: 2.5 GHz
fc1 = 3e9
fc2 = 2.5e9
tau1 = 1 / (2 * np.pi * fc1)
tau2 = 1 / (2 * np.pi * fc2)

# 兩通道交錯取樣，每個通道的取樣週期為總週期的 2 倍
T_channel = 2 * Ts      # 200 ps per channel

alpha1 = 1 - np.exp(-T_channel / tau1)
alpha2 = 1 - np.exp(-T_channel / tau2)

# 通道 1 的取樣時刻： 0 + dt1, 200 ps + dt1, 400 ps + dt1, ...
# 通道 2 的取樣時刻： 100 ps + dt2, 300 ps + dt2, 500 ps + dt2, ...
n_ch = N_samples // 2  # 每個通道的取樣點數
t_ch1 = np.arange(n_ch) * 2 * Ts + dt1
t_ch2 = np.arange(n_ch) * 2 * Ts + Ts + dt2

# 輸入信號
x_ch1 = A_signal * np.sin(2 * np.pi * f_signal * t_ch1)
x_ch2 = A_signal * np.sin(2 * np.pi * f_signal * t_ch2)

# 各通道的濾波效應 (頻寬失配)
# 一階低通濾波器的遞迴式模擬
y_ch1 = np.zeros_like(x_ch1)
y_ch2 = np.zeros_like(x_ch2)
y_ch1[0] = x_ch1[0]
y_ch2[0] = x_ch2[0]
for i in range(1, n_ch):
    y_ch1[i] = y_ch1[i - 1] + alpha1 * (x_ch1[i] - y_ch1[i - 1])
    y_ch2[i] = y_ch2[i - 1] + alpha2 * (x_ch2[i] - y_ch2[i - 1])

# 加入偏置失配
y_ch1 += offset1
y_ch2 += offset2

# 交錯合併兩個通道，形成 TI-ADC 輸出 (未校正)
y_TI_uncal = np.zeros(N_samples)
y_TI_uncal[0::2] = y_ch1  # 通道 1 放在偶數點
y_TI_uncal[1::2] = y_ch2  # 通道 2 放在奇數點

# 計算未校正的 TI-ADC 輸出 FFT / SNDR
window = np.hanning(len(y_TI_uncal))  
Y_uncal = np.fft.fft(y_TI_uncal * window, n=N_fft)
Y_uncal = Y_uncal[:N_fft // 2]
freq_axis = np.fft.fftfreq(N_fft, d=Ts)[:N_fft // 2]
mag_uncal = 20 * np.log10(np.abs(Y_uncal))

# 找到與 f_signal 最接近的 FFT bin，並計算 SNDR
bin_signal = np.argmin(np.abs(freq_axis - f_signal))
signal_power_uncal = np.abs(Y_uncal[bin_signal]) ** 2
noise_power_uncal = np.sum(np.abs(Y_uncal) ** 2) - signal_power_uncal
SNDR_uncal = 10 * np.log10(signal_power_uncal / noise_power_uncal)
print("Uncalibrated SNDR: {:.2f} dB".format(SNDR_uncal))

plt.figure(figsize=(10, 5))
plt.plot(freq_axis / 1e9, mag_uncal, label='Uncalibrated')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Uncalibrated TI-ADC Spectrum")
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------------
# 4(b) 校正：補償偏置、時間與頻寬失配
# ----------------------------------

# --- 偏置校正 ---
offset1_est = np.mean(y_ch1)
offset2_est = np.mean(y_ch2)
y_ch1_cal = y_ch1 - offset1_est
y_ch2_cal = y_ch2 - offset2_est

# --- 時間校正 ---
#   定義一個分數延遲補償函數 (線性插值)
def fractional_delay(signal, delay, T_samp):
    n = np.arange(len(signal))
    t_orig = n * T_samp
    # 要補償 -delay => np.interp 的 x 點為 t_orig + delay
    return np.interp(t_orig, t_orig + delay, signal)

y_ch1_cal = fractional_delay(y_ch1_cal, -dt1, T_channel)
y_ch2_cal = fractional_delay(y_ch2_cal, -dt2, T_channel)

# --- 頻寬校正 ---
# 依據已知的一階低通濾波器模型，在 f_signal 處計算補償增益
H1 = 1 / (1 + 1j * 2 * np.pi * f_signal * tau1)
H2 = 1 / (1 + 1j * 2 * np.pi * f_signal * tau2)
gain1 = 1 / np.abs(H1)
gain2 = 1 / np.abs(H2)

y_ch1_cal *= gain1
y_ch2_cal *= gain2

# --- 重建校正後的 TI-ADC 輸出 ---
y_TI_cal = np.zeros(N_samples)
y_TI_cal[0::2] = y_ch1_cal
y_TI_cal[1::2] = y_ch2_cal

# 計算校正後的 TI-ADC 輸出 FFT / SNDR
Y_cal = np.fft.fft(y_TI_cal * window, n=N_fft)
Y_cal = Y_cal[:N_fft // 2]
mag_cal = 20 * np.log10(np.abs(Y_cal))
signal_power_cal = np.abs(Y_cal[bin_signal]) ** 2
noise_power_cal = np.sum(np.abs(Y_cal) ** 2) - signal_power_cal
SNDR_cal = 10 * np.log10(signal_power_cal / noise_power_cal)
print("Calibrated SNDR: {:.2f} dB".format(SNDR_cal))

plt.figure(figsize=(10, 5))
plt.plot(freq_axis / 1e9, mag_cal, color='orange', label='Calibrated')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Calibrated TI-ADC Spectrum")
plt.grid(True)
plt.legend()
plt.show()


