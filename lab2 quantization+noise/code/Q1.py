import numpy as np
import matplotlib.pyplot as plt

Fs = 5e6  
F = 2e6   
N = 64 
Ts = 1 / Fs  
SNR_dB = 50  

# === continuous time signal ===
t_cont = np.linspace(0, N * Ts, 1000)  # Continuous time axis (1000 points)
x_t = np.cos(2 * np.pi * F * t_cont)   # Continuous time signal x(t)

# === discrete time signal ===
t_n = np.arange(N) * Ts  # Discrete time axis: 64 points
x_n = np.cos(2 * np.pi * F * t_n)  # Sampled signal x[n]

# === POWER signal & noise ===
signal_power = np.mean(x_n ** 2)  # P signal= 1/N (∑ (n=0~N-1) x[n]²), x[n] time domain
noise_power = signal_power / (10 ** (SNR_dB / 10))  # SNR= 10 log10(P signal/ P noise)
noise = np.sqrt(noise_power)  # avg= 0 white noise, P noise= σ² noise

# === Gaussian Noise ===
Gaussian_noise = np.random.normal(0, noise, N)  # Gaussian noise
x_noisy = x_n + Gaussian_noise  # Noisy signal

# === FFT ===
X_k = np.fft.fft(x_noisy, N)  # DFT freq amp x[k]
X_k_shifted = np.fft.fftshift(X_k)  # Move center to 0
frequencies = np.fft.fftfreq(N, Ts)  # DFT freq (0, 78.125k, ..., 2.5M)
frequencies_shifted = np.fft.fftshift(frequencies)  # Move center to 0

# === PSD  ===
df = Fs / N # bin value: 78.125k
PSD = (np.abs(X_k_shifted) ** 2 / (N * Fs)) * df  # [PSD(bin)/fs= power/hz] * bin's Hz (df)= PSD power/bin
PSD_dB = 10 * np.log10(PSD)  # dB

# === plot ===
fig, axes = plt.subplots(4, 1, figsize=(12, 18))

# **origin signal + sample point**
axes[0].plot(t_cont * 1e6, x_t, 'b-', label="Continuous Signal $x(t)$")
axes[0].stem(t_n * 1e6, x_n, 'r', markerfmt='ro', basefmt=" ", label="Sampled Points $x[n]$")
axes[0].set_title("Continuous Signal $x(t)$ and Sample Points $x[n]$")
axes[0].set_xlabel("Time (µs)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid()

# **discrete signal + sample point**
axes[1].plot(t_n * 1e6, x_n, 'r-', label="Sampled Signal $x[n]$ (connected)")  # 連線
axes[1].stem(t_n * 1e6, x_n, 'r', markerfmt='ro', basefmt=" ")  # 取樣點
axes[1].set_title("Sampled Signal $x[n]$ (Connected with Sample Points)")
axes[1].set_xlabel("Time (µs)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid()

# **signal with noise**
axes[2].plot(t_n * 1e6, x_noisy, 'k-', label="Noisy Signal")
axes[2].set_title("Noisy Sampled Signal")
axes[2].set_xlabel("Time (µs)")
axes[2].set_ylabel("Magnitude")
axes[2].legend()
axes[2].grid()

# **Power Spectral Density (PSD)**
axes[3].plot(frequencies_shifted, PSD_dB, color='green', label='PSD of Noisy Signal')
axes[3].set_title("PSD of Noisy Signal")
axes[3].set_xlabel("Frequency (MHz)")
axes[3].set_ylabel("Power (dB)")
axes[3].legend()
axes[3].grid()

plt.tight_layout()
plt.show()







