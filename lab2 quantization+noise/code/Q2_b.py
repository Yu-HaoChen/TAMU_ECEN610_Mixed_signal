import numpy as np
import matplotlib.pyplot as plt

Fs = 1.9e9                  
F = 4e8                    
N_bits = 12                 
Vref = 1                   
quantization_levels = 2 ** N_bits

def simulate_quantization(N_cycles):
    # === 1. x[n] ===
    N = int(N_cycles * Fs / F)  # N= int (period= 30) * (T0/Ts= 4.75)= 142.5, 142 (DFT only int)= 29.89cycle
    t_n = np.arange(N) / Fs     # x[n] tiem axis: n * Ts, n= 0, 1, 2, 3... N-1
    t_cont = np.linspace(0, N / Fs, 1000)  # x(t) time axis
    x_input_cont = Vref * np.cos(2 * np.pi * F * t_cont)  # x(t)
    x_input = Vref * np.cos(2 * np.pi * F * t_n)  # x[n]

    # === 2. Quantization ===
    x_quantized = np.round((x_input + Vref) * (quantization_levels - 1) / (2 * Vref)) * (2 * Vref) / (quantization_levels - 1) - Vref
    x_quantized_nodc = x_quantized - np.mean(x_quantized)
    quantization_noise = x_input - x_quantized  # Quantization error
    quantization_noise_nodc = x_input - x_quantized_nodc

    # === 3. SNR ===
    signal_power = np.mean(x_input ** 2)  # P signal
    noise_power = np.mean(quantization_noise ** 2)
    SNR = 10 * np.log10(signal_power / noise_power)

    # === 5. FFT & PSD ===
    X_k = np.fft.fft(x_quantized, N)
    X_k_shifted = np.fft.fftshift(X_k)
    frequencies = np.fft.fftfreq(N, 1 / Fs)
    frequencies_shifted = np.fft.fftshift(frequencies)
    df = Fs / N
    PSD = (np.abs(X_k_shifted) ** 2 / (N * Fs)) * df
    PSD_dB = 10 * np.log10(PSD + 1e-20)

    # === 6. plot ===
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # (1) original siganl + sample point
    axes[0].plot(t_cont * 1e9, x_input_cont, 'b-', label="Continuous Signal $x(t)$")
    axes[0].stem(t_n * 1e9, x_input, 'r', markerfmt='ro', basefmt=" ", label="Sampled Points $x[n]$")
    axes[0].set_title("Continuous Signal $x(t)$ and Sample Points $x[n]$")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid()

    # (2) discrete signal
    axes[1].stem(t_n * 1e9, x_quantized, 'r', markerfmt='ro', basefmt=" ", label="Quantized Signal $x_{quantized}[n]$")
    axes[1].set_title("Quantized Signal $x_{quantized}[n]$")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid()

    # (3) (Quantization Noise)
    axes[2].plot(t_n * 1e9, quantization_noise_nodc, 'g-', label="Quantization Noise")
    axes[2].set_title("Quantization Noise")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid()

    # (4) Power Spectral Density (PSD)
    axes[3].plot(frequencies_shifted / 1e6, PSD_dB, color='green', label='PSD of Quantized Signal')
    axes[3].set_title("PSD of Quantized Signal")
    axes[3].set_xlabel("Frequency (MHz)")
    axes[3].set_ylabel("Power (dB)")
    axes[3].set_xlim([-500, 500])  # 顯示範圍 -500 MHz 到 500 MHz
    axes[3].set_ylim([-150, 10])
    axes[3].legend()
    axes[3].grid()

    plt.tight_layout()
    plt.show()

    return SNR

# === 7.示 SNR ===
snr_30_cycles = simulate_quantization(30)
print(f"\n=== SNR for 29.8 Cycles ===")
print(f"SNR = {snr_30_cycles:.2f} dB")