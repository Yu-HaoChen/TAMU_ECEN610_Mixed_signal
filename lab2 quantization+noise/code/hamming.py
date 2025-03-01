import numpy as np
import matplotlib.pyplot as plt

Fs = 5e6                
F = 2e6                 
N = 64                  
Ts = 1 / Fs             
SNR_dB = 50             

# === 1.(Sampled Signal) ===
t_n = np.arange(N) * Ts  # discrete time axis
x_n = np.cos(2 * np.pi * F * t_n)  # sample signal

# === 2.Gaussian noise ===
signal_power = np.mean(x_n ** 2)   # signal power
noise_power = signal_power / (10 ** (SNR_dB / 10))  # noise power
noise_std = np.sqrt(noise_power)   # noise standard
Gaussian_noise = np.random.normal(0, noise_std, N)  # Gaussian Noise
x_noisy = x_n + Gaussian_noise      # signal with noise

# === 3. Hamming Window ===
hanning_window = np.hamming(N)  # Hanning Window
x_windowed = x_noisy * hanning_window  # signal with noise + Window

# === 4. FFT & freq axis ===
X_k = np.fft.fft(x_windowed, N)  # FFT Spectrum
X_k_shifted = np.fft.fftshift(X_k)  # center 0
frequencies = np.fft.fftfreq(N, Ts)  # FFT freq axis (0, 78.125k, ... 2.5M)
frequencies_shifted = np.fft.fftshift(frequencies)  # center 0

# === 5. Power Spectral Density (PSD) ===
df = Fs / N
PSD = (np.abs(X_k_shifted) ** 2 / (N * Fs)) * df  # PSD
PSD_dB = 10 * np.log10(PSD)  # dB 單位

# === 6. plot ===
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# origin signal
axes[0].plot(t_n * 1e6, x_n, 'b-', label="Sampled Signal $x[n]$")
axes[0].stem(t_n * 1e6, x_n, 'r', markerfmt='ro', basefmt=" ", label="Sample Points $x[n]$")
axes[0].set_title("Sampled Signal $x[n]$")
axes[0].set_xlabel("Time (µs)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid()

# Hanning Window siganl
axes[1].plot(t_n * 1e6, x_windowed, 'g-', label="Windowed Signal $x_{windowed}[n]$")
axes[1].set_title("Hamming Windowed Signal $x_{windowed}[n]$")
axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid()

# Power Spectral Density (PSD)
axes[2].plot(frequencies_shifted / 1e6, PSD_dB, color='green', label='PSD (Hamming Window)')
axes[2].set_title("PSD of Hamming Windowed Signal")
axes[2].set_xlabel("Frequency (MHz)")
axes[2].set_ylabel("Power (dB)")
axes[2].set_xlim([-2.5, 2.5])  # 顯示 -2.5 MHz 到 2.5 MHz
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()

