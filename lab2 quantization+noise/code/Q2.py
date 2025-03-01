import numpy as np
import matplotlib.pyplot as plt

Fs = 6e8                  
F = 2e8                   
N_bits = 12                
Vref = 1                   
quantization_levels = 2 ** N_bits

def simulate_quantization(N_cycles):
    # === 1. x[n] ===
    N = int(N_cycles * Fs / F)  # N= int (period) * (T0/Ts= sample point in one T0)
    t_n = np.arange(N) / Fs     # x[n] tiem axis: n * Ts, n= 0, 1, 2, 3... N-1
    t_cont = np.linspace(0, N / Fs, 1000)  # x(t) time axis
    x_input_cont = Vref * np.sin(2 * np.pi * F * t_cont)  # x(t)
    x_input = Vref * np.sin(2 * np.pi * F * t_n)  # x[n]

    # === 2. Quantization ===
    x_quantized_con = np.round((x_input_cont + Vref) * (quantization_levels - 1) / (2 * Vref)) * (2 * Vref) / (quantization_levels - 1) - Vref
    x_quantized = np.round((x_input + Vref) * (quantization_levels - 1) / (2 * Vref)) * (2 * Vref) / (quantization_levels - 1) - Vref
    x_quantized_nodc = x_quantized - np.mean(x_quantized)
    quantization_noise = x_input - x_quantized  # Quantization error
    print(x_quantized)
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

    # (1) original signal + sample point + quantized points
    axes[0].plot(t_cont * 1e9, x_input_cont, 'b-', label="Continuous Signal $x(t)$")
    axes[0].stem(t_n * 1e9, x_input, 'r', markerfmt='ro', basefmt=" ", label="Sampled Points $x[n]$")
    axes[0].plot(t_n * 1e9, x_quantized, 'g', label="Quantized Points $x_{quantized}[n]$")
    axes[0].set_title("Continuous Signal $x(t)$, Sampled Points $x[n]$, and Quantized Points $x_{quantized}[n]$")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid()

    # (2) Continuous Quantized Signal
    axes[1].plot(t_cont * 1e9, x_quantized_con, 'black', label="Continuous Quantized Signal $x_{quantized\_con}(t)$")
    axes[1].set_title("Continuous Quantized Signal $x_{quantized\_con}(t)$")
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
    axes[3].legend()
    axes[3].grid()

    plt.tight_layout()
    plt.show()

    return SNR

# === 7.示 SNR ===
snr_30_cycles = simulate_quantization(30)
print(f"\n=== SNR for 30 Cycles ===")
print(f"SNR = {snr_30_cycles:.2f} dB")