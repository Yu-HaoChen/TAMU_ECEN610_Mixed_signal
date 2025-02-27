import numpy as np
import matplotlib.pyplot as plt

Fs = 1.9e9                   
F = 4e8                     
N_bits = 12                 
Vref = 1                    
quantization_levels = 2 ** N_bits

def simulate_quantization(N_cycles, use_window=False):
    # === 1. x[n] ===
    N = int(N_cycles * Fs / F)  # N= int (period) * (T0/Ts= sample point in one T0)
    t_n = np.arange(N) / Fs     # x[n] time axis: n * Ts, n= 0, 1, 2, 3... N-1
    t_cont = np.linspace(0, N / Fs, 1000)  # x(t) time axis
    x_input_cont = Vref * np.cos(2 * np.pi * F * t_cont)  # x(t)
    x_input = Vref * np.cos(2 * np.pi * F * t_n)  # x[n]

    # === 2. Apply Hanning Window (Optional) ===
    if use_window:
        window = np.hanning(N)
        x_windowed = x_input * window
    else:
        x_windowed = x_input

    # === 3. Quantization ===
    x_quantized = np.round((x_windowed + Vref) * (quantization_levels - 1) / (2 * Vref)) * (2 * Vref) / (quantization_levels - 1) - Vref
    x_quantized_nodc = x_quantized - np.mean(x_quantized)
    quantization_noise = x_windowed - x_quantized  # Quantization error
    quantization_noise_nodc = x_windowed - x_quantized_nodc

    # === 4. SNR ===
    signal_power = np.mean(x_windowed ** 2)  # P signal
    noise_power = np.mean(quantization_noise ** 2)
    SNR = 10 * np.log10(signal_power / noise_power)

    # === 5. FFT & PSD ===
    X_k = np.fft.fft(x_quantized_nodc, N)
    X_k_shifted = np.fft.fftshift(X_k)
    frequencies = np.fft.fftfreq(N, 1 / Fs)
    frequencies_shifted = np.fft.fftshift(frequencies)
    df = Fs / N
    PSD = (np.abs(X_k_shifted) ** 2 / (N * Fs)) * df
    PSD_dB = 10 * np.log10(PSD + 1e-20)

    # === 6. Plot ===
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # (1) Original Signal + Sample Points
    axes[0].plot(t_cont * 1e9, x_input_cont, 'b-', label="Continuous Signal $x(t)$")
    axes[0].stem(t_n * 1e9, x_input, 'r', markerfmt='ro', basefmt=" ", label="Sampled Points $x[n]$")
    axes[0].set_title("Continuous Signal $x(t)$ and Sample Points $x[n]$")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid()

    # (2) Quantized Signal
    axes[1].stem(t_n * 1e9, x_quantized, 'r', markerfmt='ro', basefmt=" ", label="Quantized Signal $x_{quantized}[n]$")
    axes[1].set_title("Quantized Signal $x_{quantized}[n]$" + (" (with Hanning Window)" if use_window else ""))
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid()

    # (3) Quantization Noise
    axes[2].plot(t_n * 1e9, quantization_noise_nodc, 'g-', label="Quantization Noise")
    axes[2].set_title("Quantization Noise")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid()

    # (4) Power Spectral Density (PSD)
    axes[3].plot(frequencies_shifted / 1e6, PSD_dB, color='green', label='PSD of Quantized Signal')
    axes[3].set_title("PSD of Quantized Signal" + (" (with Hanning Window)" if use_window else ""))
    axes[3].set_xlabel("Frequency (MHz)")
    axes[3].set_ylabel("Power (dB)")
    axes[3].set_xlim([-500, 500])  # 顯示範圍 -500 MHz 到 500 MHz
    axes[3].set_ylim([-150, 10])
    axes[3].legend()
    axes[3].grid()

    plt.tight_layout()
    plt.show()

    # === 7. Print PSD ===
    print("\n=== PSD Values ===")
    for freq, psd_val in zip(frequencies_shifted / 1e6, PSD_dB):
        print(f"Frequency: {freq:.2f} MHz, PSD: {psd_val:.2f} dB")

    return SNR

# === 8. SNR without Window ===
snr_30_cycles_no_window = simulate_quantization(30, use_window=False)
print(f"\n=== SNR for 30 Cycles (No Window) ===")
print(f"SNR = {snr_30_cycles_no_window:.2f} dB")

# === 9. SNR with Hanning Window ===
snr_30_cycles_window = simulate_quantization(30, use_window=True)
print(f"\n=== SNR for 30 Cycles (Hanning Window) ===")
print(f"SNR = {snr_30_cycles_window:.2f} dB")

