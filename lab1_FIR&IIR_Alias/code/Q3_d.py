import numpy as np
import matplotlib.pyplot as plt

Fs = 1e9  
F1 = 200e6  
F2 = 400e6  
N = 64  
Ts = 1 / Fs  

# === Discrete-Time Signal ===
t_n = np.arange(N) * Ts  # 64 points
y_n = np.cos(2 * np.pi * F1 * t_n) + np.cos(2 * np.pi * F2 * t_n)  # original signal

# === Blackman ===
blackman_window = np.blackman(N)  # generate blackman
y_n_windowed = y_n * blackman_window  # adding blackman

# === FFT ===
Y_k = np.fft.fft(y_n, N)  # FFT
Y_k_windowed = np.fft.fft(y_n_windowed, N)  # Black FFT

# === Freq axis ===
frequencies = np.fft.fftfreq(N, Ts)
frequencies_shifted = np.fft.fftshift(frequencies)

# ===  0 Hz center ===
Y_k_shifted = np.fft.fftshift(Y_k)
Y_k_windowed_shifted = np.fft.fftshift(Y_k_windowed)

# === plot ===
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# *no black*
axes[0].plot(frequencies_shifted / 1e6, np.abs(Y_k_shifted) / np.max(np.abs(Y_k_shifted)), label="Original FFT")
axes[0].set_title("FFT Magnitude Spectrum (No Window)")
axes[0].set_xlabel("Frequency (MHz)")
axes[0].set_ylabel("Magnitude")
axes[0].grid()
axes[0].legend()

# **black**
axes[1].plot(frequencies_shifted / 1e6, np.abs(Y_k_windowed_shifted) / np.max(np.abs(Y_k_windowed_shifted)), label="Blackman Window FFT", color='r')
axes[1].set_title("FFT Magnitude Spectrum (With Blackman Window)")
axes[1].set_xlabel("Frequency (MHz)")
axes[1].set_ylabel("Magnitude")
axes[1].grid()
axes[1].legend()

plt.tight_layout()
plt.show()
