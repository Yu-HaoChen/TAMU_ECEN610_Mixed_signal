import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# === Define FIR and IIR Filters ===

# FIR filter: H(z) = 1 + z^(-1) + z^(-2) + z^(-3) + z^(-4)
fir_b = [1, 1, 1, 1, 1]  # FIR numerator coefficients (Zero)
fir_a = [1]  # FIR filter denominator (only 1 for FIR) (no pole)

# IIR filter: H(z) = (1 + z^(-1)) / (1 - z^(-1))
iir_b = [1, 1]  # IIR numerator coefficients (zero)
iir_a = [1, -1]  # IIR denominator coefficients (poles)

# === Compute Frequency Response ===
w_fir, h_fir = signal.freqz(fir_b, fir_a, worN=1024) # w, h = signal.freqz(b(分子係數), a(分母係數), worN=1024)
w_iir, h_iir = signal.freqz(iir_b, iir_a, worN=1024) # w(0 ~ π rad/sample) h(Complex values, including amplitude & phase)

# === Compute Poles and Zeros ===
zeros_fir, poles_fir, _ = signal.tf2zpk(fir_b, fir_a) # zeros, poles, gain = signal.tf2zpk(b, a)
zeros_iir, poles_iir, _ = signal.tf2zpk(iir_b, iir_a)

#----------------------------------------------------------------------------------------------------------------------

# === Plot Frequency Response and Pole-Zero Plot ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- FIR Frequency Response ---
# x-axis: [normalized w(0 ~ π rad/sample)/ π= 0 ~ 1]  y-axis: 20log (Complex values)
axes[0, 0].plot(w_fir / np.pi, 20 * np.log10(abs(h_fir)), label="FIR", color="blue") 
axes[0, 0].set_title("FIR Filter Frequency Response")
axes[0, 0].set_xlabel("Normalized Frequency (π rad/sample)")
axes[0, 0].set_ylabel("Magnitude (dB)")
axes[0, 0].grid()
axes[0, 0].legend()

# --- IIR Frequency Response ---
axes[0, 1].plot(w_iir / np.pi, 20 * np.log10(abs(h_iir)), label="IIR", color="red")
axes[0, 1].set_title("IIR Filter Frequency Response")
axes[0, 1].set_xlabel("Normalized Frequency (π rad/sample)")
axes[0, 1].set_ylabel("Magnitude (dB)")
axes[0, 1].grid()
axes[0, 1].legend()

# --- FIR Pole-Zero Plot ---
axes[1, 0].scatter(np.real(zeros_fir), np.imag(zeros_fir), color='blue', label="Zeros", marker='o')
axes[1, 0].scatter(np.real(poles_fir), np.imag(poles_fir), color='red', label="Poles", marker='x')
unit_circle = plt.Circle((0, 0), 1, color='gray', linestyle='dashed', fill=False) # plt.circle ((圓心), 半徑)
axes[1, 0].add_patch(unit_circle)
axes[1, 0].set_xlim([-1.5, 1.5]) # set x-axis more than 1
axes[1, 0].set_ylim([-1.5, 1.5])
axes[1, 0].set_xlabel("Real")
axes[1, 0].set_ylabel("Imaginary")
axes[1, 0].set_title("Poles and Zeros of FIR Filter")
axes[1, 0].axhline(0, color='black', linewidth=0.5)
axes[1, 0].axvline(0, color='black', linewidth=0.5)
axes[1, 0].grid()
axes[1, 0].legend()

# --- IIR Pole-Zero Plot ---
axes[1, 1].scatter(np.real(zeros_iir), np.imag(zeros_iir), color='blue', label="Zeros", marker='o')
axes[1, 1].scatter(np.real(poles_iir), np.imag(poles_iir), color='red', label="Poles", marker='x')
unit_circle = plt.Circle((0, 0), 1, color='gray', linestyle='dashed', fill=False)
axes[1, 1].add_patch(unit_circle)
axes[1, 1].set_xlim([-1.5, 1.5])
axes[1, 1].set_ylim([-1.5, 1.5])
axes[1, 1].set_xlabel("Real")
axes[1, 1].set_ylabel("Imaginary")
axes[1, 1].set_title("Poles and Zeros of IIR Filter")
axes[1, 1].axhline(0, color='black', linewidth=0.5)
axes[1, 1].axvline(0, color='black', linewidth=0.5)
axes[1, 1].grid()
axes[1, 1].legend()

# === Overall Figure Title ===
fig.suptitle("Frequency Response and Pole-Zero Plots of FIR and IIR Filters", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
