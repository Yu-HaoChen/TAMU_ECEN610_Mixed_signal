import numpy as np
import matplotlib.pyplot as plt

Fs = 800e6
F1 = 300e6  
T = 10 / F1  # Running time for 10 cycle (10Ts)
Ts = 1 / Fs  # sample rate  1.25e-9 sec

# === Continuous time signal ===
t_cont = np.linspace(0, T, 1000)  # Continuous time 0 ~ T make up with 1000points
x_cont = np.cos(2 * np.pi * F1 * t_cont)  # Original signal Amp cos()

# === Sampled signal ===
t_sampled = np.arange(0, T-Ts, Ts)  # discrete time 0 ~ T sample every Ts
x_sampled = np.cos(2 * np.pi * F1 * t_sampled)

# === Sinc interpolation function ===
def sinc_interp(xn, tn, t_interp): #(sample point, sample point's time, reconstruct time)
    """ sinc to get original signal """
    Ts = tn[1] - tn[0]  # sample time= xn[i] - xn[i-1]= Ts= 1/Fs
    return np.sum(xn * np.sinc((t_interp[:, None] - tn) / Ts), axis=1)
# multiplies the sinc function weight by each sampled value xn, sinc, axis=1 (horizontal summation)

# Reconstructed signal (Ts sample)
x_reconstructed_Ts = sinc_interp(x_sampled, t_sampled, t_cont)

#**********************************************************************************************************************

# Shifted sample (Ts/2)
t_sampled_shifted = np.arange(Ts/2, T-(Ts/2), Ts)  # Sample points shifted by Ts/2
x_sampled_shifted = np.cos(2 * np.pi * F1 * t_sampled_shifted)  # Shifted sampled values
x_reconstructed_Ts_half = sinc_interp(x_sampled_shifted, t_sampled_shifted, t_cont)

# === MSE ===
mse_Ts = np.mean((x_reconstructed_Ts - x_cont) ** 2)
mse_Ts_half = np.mean((x_reconstructed_Ts_half - x_cont) ** 2)

#----------------------------------------------------------------------------------------------------------------------

# === plot ===
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

#**original signal + sample point**
axes[0].plot(t_cont * 1e9, x_cont, 'b-', label="Original Continuous Signal")
axes[0].stem(t_sampled * 1e9, x_sampled, 'r', markerfmt='ro', label="Sampled Signal at Ts", basefmt=" ")
axes[0].set_title("Original Signal and Sampling at Ts", loc="left")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid()

#**reconsturct signal**
axes[1].plot(t_cont * 1e9, x_cont, 'b-', label="Original Continuous Signal")
axes[1].plot(t_cont * 1e9, x_reconstructed_Ts, 'g--', label="Reconstructed Signal (Ts)")
axes[1].set_title(f"Reconstructed Signal from Samples at Ts (MSE: {mse_Ts:.4e})", loc="left")
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid()

#**Shifted signal**
axes[2].plot(t_cont * 1e9, x_cont, 'b-', label="Original Continuous Signal")
axes[2].plot(t_cont * 1e9, x_reconstructed_Ts_half, 'orange', linestyle='--', label="Reconstructed Signal (Ts/2)")
axes[2].set_title(f"Reconstructed Signal from Samples at Ts/2 (MSE: {mse_Ts_half:.4e})", loc="left")
axes[2].set_xlabel("Time (ns)")
axes[2].set_ylabel("Amplitude")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()


