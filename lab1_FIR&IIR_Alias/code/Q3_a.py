import numpy as np
import matplotlib.pyplot as plt

Fs = 5e6   # Sampling Frequency
F = 2e6    # Signal Frequency
N = 64     # FFT Points
Ts = 1 / Fs  # Sampling Period

# === Continuous Time Signal ===
t_cont = np.linspace(0, N * Ts, 1000)  # Continuous 1000 points
x_cont = np.cos(2 * np.pi * F * t_cont)  # Original signal

# === Sampled Signal (Discrete Time) ===
t_n = np.arange(N) * Ts  # 64 sample points
x_n = np.cos(2 * np.pi * F * t_n)  # Sampled cosine wave

# === FFT Calculation ===
X_k = np.fft.fft(x_n, N)  # 64-point FFT
frequencies = np.fft.fftfreq(N, Ts)  # Corresponding frequency axis fftfreq(N, time)
X_k_shifted = np.fft.fftshift(X_k)   # Center zero frequency
frequencies_shifted = np.fft.fftshift(frequencies) / 1e6  # Convert to MHz

# === Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Time Domain Plot
axes[0].plot(t_cont * 1e6, x_cont, label="Continuous Signal")
axes[0].stem(t_n * 1e6, x_n, 'r', markerfmt='ro', label="Sampled Points", basefmt=" ")
axes[0].set_title("3.a. Time Domain Sample of Cosine Function")
axes[0].set_xlabel("Time (µs)")
axes[0].set_ylabel("Magnitude")
axes[0].legend()
axes[0].grid()

# Frequency Domain Plot
axes[1].plot(frequencies_shifted, np.abs(X_k_shifted) / np.max(np.abs(X_k_shifted)))
axes[1].set_title("3.a. Frequency Domain of Cosine Function")
axes[1].set_xlabel("Frequency (MHz)")
axes[1].set_ylabel("Magnitude (Normalized)")
axes[1].grid()

plt.tight_layout()
plt.show()

