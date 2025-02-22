import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# === Fin ===
Fs = 500e6  
F1 = 300e6  
F2 = 800e6  
N = 128

t = np.linspace(0, N/Fs, N, endpoint=False)  # contionus time
n = np.arange(N)  # discrete time

# === contimuous time signal ===
x1_t = np.cos(2 * np.pi * F1 * t)  # contionus time x1(t)
x2_t = np.cos(2 * np.pi * F2 * t)  # contionus time x2(t)

# === Fa freq ===
k1 = round(F1 / Fs) 
Fa1 = abs(F1 - k1 * Fs) 
k2 = round(F2 / Fs)  
Fa2 = abs(F2 - k2 * Fs)  

x1_n = np.cos(2 * np.pi * Fa1 * n / Fs)  # x1(n)
x2_n = np.cos(2 * np.pi * Fa2 * n / Fs)  # x2(n)

# ===  FFT  ===
X1_f = np.abs(fft(x1_n))[:N//2]  
X2_f = np.abs(fft(x2_n))[:N//2]
freqs = np.fft.fftfreq(N, d=1/Fs)[:N//2]  # freq ( Nyquist range)

# === plot ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 左上：F1 時域
axes[0, 0].plot(t * 1e9, x1_t, color='blue')
axes[0, 0].set_title(f"Time Domain Signal (F1 = {F1/1e6} MHz)")
axes[0, 0].set_xlabel("Time (ns)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid()

# 右上：F2 時域
axes[0, 1].plot(t * 1e9, x2_t, color='red')
axes[0, 1].set_title(f"Time Domain Signal (F2 = {F2/1e6} MHz)")
axes[0, 1].set_xlabel("Time (ns)")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].grid()

# 左下：FFT 頻譜（alias 頻率）
axes[1, 0].stem(freqs/1e6, X1_f, linefmt='b-', markerfmt='bo', basefmt=" ", label=f"Alias of F1 = {Fa1/1e6} MHz")
axes[1, 0].stem(freqs/1e6, X2_f, linefmt='r--', markerfmt='ro', basefmt=" ", label=f"Alias of F2 = {Fa2/1e6} MHz")
axes[1, 0].set_title("Frequency Domain (Aliased Frequencies)")
axes[1, 0].set_xlabel("Frequency (MHz)")
axes[1, 0].set_ylabel("Magnitude")
axes[1, 0].legend()
axes[1, 0].grid()

# 右下：Alias 時域
axes[1, 1].stem(n, x1_n, linefmt='b-', markerfmt='bo', basefmt=" ", label=f"Aliased x1(n) - {Fa1/1e6} MHz")
axes[1, 1].stem(n, x2_n, linefmt='r--', markerfmt='ro', basefmt=" ", label=f"Aliased x2(n) - {Fa2/1e6} MHz")
axes[1, 1].set_title("Aliased Time Domain Signals")
axes[1, 1].set_xlabel("Sample index n")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()
