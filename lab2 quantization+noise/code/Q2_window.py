import numpy as np
import matplotlib.pyplot as plt

Fs = 5e8       
F = 2e8        
N_bits = 12    
Vref = 1       

quantization_levels = 2 ** N_bits

def simulate_quantization(N_cycles):
    N = int(N_cycles * Fs / F)        
    t_n = np.arange(N) / Fs           
    t_cont = np.linspace(0, N / Fs, 1000)  
    
    x_input_cont = Vref * np.sin(2 * np.pi * F * t_cont)
    x_input = Vref * np.sin(2 * np.pi * F * t_n)

    x_quantized = np.round((x_input + Vref) * (quantization_levels - 1) / (2 * Vref)) \
                  * (2 * Vref) / (quantization_levels - 1) - Vref
    quantization_noise = x_input - x_quantized


    signal_power = np.mean(x_input ** 2)
    noise_power = np.mean(quantization_noise ** 2)
    SNR = 10 * np.log10(signal_power / noise_power)

    window = np.hanning(N)           
    x_windowed = x_quantized * window  

    X_k = np.fft.fft(x_windowed, N)
    X_k_shifted = np.fft.fftshift(X_k)

    frequencies = np.fft.fftfreq(N, 1 / Fs)
    frequencies_shifted = np.fft.fftshift(frequencies)

    df = Fs / N
    PSD = (np.abs(X_k_shifted) ** 2 / (N * Fs)) * df
    PSD_dB = 10 * np.log10(PSD + 1e-20)  

    fig, axes = plt.subplots(5, 1, figsize=(12, 18))

    # (1) 連續訊號 + 取樣點
    axes[0].plot(t_cont * 1e9, x_input_cont, 'b-', label="Continuous Signal $x(t)$")
    axes[0].stem(t_n * 1e9, x_input, 'r', markerfmt='ro', basefmt=" ",
                 label="Sampled Points $x[n]$")
    axes[0].set_title("1) Continuous Signal $x(t)$ and Sampled Points $x[n]$")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    # (2) 量化後訊號
    axes[1].stem(t_n * 1e9, x_quantized, 'r', markerfmt='ro', basefmt=" ",
                 label="Quantized Signal $x_{quantized}[n]$")
    axes[1].set_title("2) Quantized Signal $x_{quantized}[n]$")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    # (3) 加了 Hann window 的訊號 (時域)
    axes[2].stem(t_n * 1e9, x_windowed, 'm', markerfmt='mo', basefmt=" ",
                 label="Windowed Signal $x_{windowed}[n]$")
    axes[2].set_title("3) Windowed Signal (Hann) in Time Domain")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True)

    # (4) 量化雜訊
    axes[3].plot(t_n * 1e9, quantization_noise, 'g-', label="Quantization Noise")
    axes[3].set_title("4) Quantization Noise")
    axes[3].set_xlabel("Time (ns)")
    axes[3].set_ylabel("Amplitude")
    axes[3].legend()
    axes[3].grid(True)

    # (5) PSD (Hann window)
    axes[4].plot(frequencies_shifted / 1e6, PSD_dB, 'g', label='PSD (Hann-windowed)')
    axes[4].set_title("5) PSD of Quantized Signal (Hanning Window)")
    axes[4].set_xlabel("Frequency (MHz)")
    axes[4].set_ylabel("Power (dB)")
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()
    plt.show()

    return SNR

snr_30_cycles = simulate_quantization(30)
print(f"\n=== SNR for 30 Cycles ===")
print(f"SNR = {snr_30_cycles:.2f} dB")

