import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
Fs = 0.5e9   # Sampling Frequency (1 GHz)
F1 = 200e6  # Frequency component 1 (200 MHz)
F2 = 400e6  # Frequency component 2 (400 MHz)
N = 64      # FFT Points
Ts = 1 / Fs # Sampling Period

# === Continuous Time Signal ===
t_cont = np.linspace(0, N * Ts, 1000)  # 1000-point continuous signal
y_cont = np.cos(2 * np.pi * F1 * t_cont) + np.cos(2 * np.pi * F2 * t_cont)

# === Sampled Signal (Discrete Time) ===
t_n = np.arange(N) * Ts  # 64 sample points
y_n = np.cos(2 * np.pi * F1 * t_n) + np.cos(2 * np.pi * F2 * t_n)  # Sampled signal

# === FFT Calculation ===
Y_k = np.fft.fft(y_n, N)  # 64-point FFT
frequencies = np.fft.fftfreq(N, Ts)  # Corresponding frequency axis
Y_k_shifted = np.fft.fftshift(Y_k)   # Center zero frequency
frequencies_shifted = np.fft.fftshift(frequencies) / 1e6  # Convert to MHz

# === Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Time Domain Plot
axes[0].plot(t_cont * 1e9, y_cont, label="Continuous Signal")
axes[0].stem(t_n * 1e9, y_n, 'r', markerfmt='ro', label="Sampled Points", basefmt=" ")
axes[0].set_title("Time Domain Sample of Cosine Function", loc="left")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("Magnitude")
axes[0].legend()
axes[0].grid()

# Frequency Domain Plot
axes[1].plot(frequencies_shifted, np.abs(Y_k_shifted) / np.max(np.abs(Y_k_shifted)))
axes[1].set_title("Frequency Domain of Cosine Function", loc="left")
axes[1].set_xlabel("Frequency (MHz)")
axes[1].set_ylabel("Magnitude (Normalized)")
axes[1].grid()

plt.tight_layout()
plt.show()
