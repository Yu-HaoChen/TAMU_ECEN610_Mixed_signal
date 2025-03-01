import numpy as np
import matplotlib.pyplot as plt

Fs = 5e8       # 取樣頻率 (Hz)
F = 2e8        # 輸入信號頻率 (Hz)
N_bits = 12    # 量化位元數
Vref = 1       # 參考電壓 (量化範圍: -Vref ~ +Vref)
quantization_levels = 2 ** N_bits

def simulate_quantization_5plots(N_cycles, p_noise_total=1.585e-4):
    """
    中文程式 + 英文註解:
    This function produces 5 subplots:
      (1) Continuous Signal + Sampled Points
      (2) Quantized Signal
      (3) Windowed Signal (Hann) in Time Domain
      (4) Quantization Noise (with extra Gaussian)
      (5) PSD of Quantized Signal (Hann Window)

    Then, we compute the time-domain SNR:
      SNR = 10 * log10( mean(x_input^2) / mean(quantization_noise^2) )
    and display it in the terminal.
    """

    # 1) Generate discrete-time signal x[n] for N integer cycles
    N = int(N_cycles * Fs / F)
    t_n = np.arange(N) / Fs
    t_cont = np.linspace(0, N / Fs, 1000)  # For continuous plotting

    # Create a clean sinusoidal input signal
    x_input_cont = Vref * np.sin(2 * np.pi * F * t_cont)  # continuous
    x_input = Vref * np.sin(2 * np.pi * F * t_n)          # discrete

    # 2) Quantize the clean signal
    x_quantized = np.round((x_input + Vref) * (quantization_levels - 1) / (2 * Vref)) \
                  * (2 * Vref) / (quantization_levels - 1) - Vref

    # 3) Compute original quantization error power
    p_quant_error = np.mean((x_input - x_quantized)**2)

    # 4) Compute p_noise = p_noise_total - p_quant_error
    p_noise = p_noise_total - p_quant_error
    if p_noise < 0:
        p_noise = 0.0
    sigma = np.sqrt(p_noise)

    # Generate Gaussian noise with variance = p_noise
    G_noise = np.random.normal(0, sigma, size=x_quantized.shape)

    # Final quantization noise = (x_input - x_quantized) + G_noise
    quantization_noise = (x_input - x_quantized) + G_noise
    quantization_noise_nodc = quantization_noise - np.mean(quantization_noise)

    # 5) Create a Hann window for the quantized signal
    window = np.hanning(N)
    x_windowed = x_quantized * window

    # 6) Compute FFT & PSD (Hann-windowed)
    X_k = np.fft.fft(x_windowed, N)
    X_k_shifted = np.fft.fftshift(X_k)
    freqs = np.fft.fftfreq(N, 1 / Fs)
    freqs_shifted = np.fft.fftshift(freqs)

    df = Fs / N
    psd = (np.abs(X_k_shifted) ** 2 / (N * Fs)) * df
    psd_dB = 10 * np.log10(psd + 1e-20)

    # 7) Plot the same 5 subplots as before
    fig, axes = plt.subplots(5, 1, figsize=(12, 18))

    # (1) Continuous Signal + Sampled Points
    axes[0].plot(t_cont * 1e9, x_input_cont, 'b-', label="Continuous Signal $x(t)$")
    axes[0].stem(t_n * 1e9, x_input, 'r', markerfmt='ro', basefmt=" ",
                 label="Sampled Points $x[n]$")
    axes[0].set_title("1) Continuous Signal $x(t)$ and Sampled Points $x[n]$")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    # (2) Quantized Signal
    axes[1].stem(t_n * 1e9, x_quantized, 'r', markerfmt='ro', basefmt=" ",
                 label="Quantized Signal $x_{quantized}[n]$")
    axes[1].set_title("2) Quantized Signal $x_{quantized}[n]$")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    # (3) Windowed Signal (Hann) in Time Domain
    axes[2].stem(t_n * 1e9, x_windowed, 'm', markerfmt='mo', basefmt=" ",
                 label="Windowed Signal $x_{windowed}[n]$")
    axes[2].set_title("3) Windowed Signal (Hann) in Time Domain")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True)

    # (4) Quantization Noise (with extra Gaussian)
    axes[3].plot(t_n * 1e9, quantization_noise_nodc, 'g-',
                 label="Quantization Noise (no DC)")
    axes[3].set_title("4) Quantization Noise (with extra Gaussian)")
    axes[3].set_xlabel("Time (ns)")
    axes[3].set_ylabel("Amplitude")
    axes[3].legend()
    axes[3].grid(True)

    # (5) PSD of Quantized Signal (Hann Window)
    axes[4].plot(freqs_shifted / 1e6, psd_dB, 'g', label='PSD (Hann-windowed)')
    axes[4].set_title("5) PSD of Quantized Signal (Hanning Window)")
    axes[4].set_xlabel("Frequency (MHz)")
    axes[4].set_ylabel("Power (dB)")
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()
    plt.show()

    # 8) Compute and print time-domain SNR
    #    SNR = 10 * log10( mean(x_input^2) / mean(quantization_noise^2) )
    signal_power = np.mean(x_input**2)
    noise_power = np.mean(quantization_noise**2)
    snr_val = 10 * np.log10(signal_power / noise_power)

    print(f"\n=== Measured SNR ===")
    print(f"SNR = {snr_val:.2f} dB")

    # Print some diagnostic info
    print(f"p_quant_error = {p_quant_error:.6e}")
    print(f"p_noise_total = {p_noise_total:.6e}")
    print(f"p_noise       = {p_noise:.6e} ( = p_noise_total - p_quant_error )")

# === Example usage ===
simulate_quantization_5plots(N_cycles=30, p_noise_total=1.585e-4)


