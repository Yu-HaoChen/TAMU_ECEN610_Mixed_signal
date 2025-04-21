"""
adc_calib_numpy.py
------------------
Pure‑NumPy demo: create an 8‑bit Nyquist‑rate ADC,
inject cubic static non‑linearity, then correct it
via bit‑wise least‑squares (pseudo‑inverse).

Outputs
-------
1) Console JSON  —  SNDR / SFDR before & after calibration
2) FFT_before.png / FFT_after.png  —  PSD plots
"""

import numpy as np
import matplotlib.pyplot as plt
import json

# =============== User‑Config =======================
N_BITS = 8            # ADC resolution
FS     = 1.0e6        # sample rate  (Hz)
FIN    = 50e3         # input tone   (Hz)
N_SMP  = 4096         # number of samples (use power of 2)

# Static cubic distortion coefficients
GAIN   = 0.9          # linear gain error (<1 means attenuation)
K3     = 3e-4         # cubic‑term strength (higher ⇒ worse SNDR)
# ===================================================

# ---------- 1. Generate test waveform --------------
t = np.arange(N_SMP) / FS
ideal_v = np.sin(2 * np.pi * FIN * t)        # full‑scale sine, range [-1, 1]

# ---------- 2. Ideal ADC quantization --------------
def adc_quantize(v):
    full = 2 ** N_BITS - 1
    return np.round((v + 1) * full / 2).astype(int)

code_ideal = adc_quantize(ideal_v)

# ---------- 3. Inject static non‑linearity ---------
def static_nonlinear(code):
    """Apply gain + cubic term, then requantize."""
    full = 2 ** N_BITS - 1
    v = (code / full) * 2 - 1                 # back to analog domain [-1,1]
    v_bad = GAIN * v + K3 * v**3
    code_bad = np.clip(np.round((v_bad + 1) * full / 2), 0, full).astype(int)
    return code_bad

code_bad = static_nonlinear(code_ideal)

# ---------- 4. Build bit‑wise design matrix --------
#   bits : shape (N_SMP, N_BITS), each column is one bit (LSB .. MSB)
bits = ((code_bad[:, None] >> np.arange(N_BITS)) & 1).astype(float)

# ---------- 5. Least‑Squares calibration ----------
# Solve   bits · w ≈ code_ideal   in L2 sense
w = np.linalg.pinv(bits) @ code_ideal         # pseudo‑inverse
code_cal = np.round(bits @ w).astype(int)     # calibrated integer codes

# ---------- 6. Metrics: SNDR & SFDR ---------------
def sndr_sfdr(codes):
    """Compute SNDR & SFDR from integer codes"""
    # remove DC, apply Hann window
    x = codes - np.mean(codes)
    X = np.fft.rfft(x * np.hanning(len(x)))
    mag = np.abs(X)

    # find signal bin (max magnitude excluding DC)
    sig_bin = np.argmax(mag[1:]) + 1
    sig_pow = mag[sig_bin] ** 2
    noise_pow = np.sum(mag**2) - sig_pow

    sndr = 10 * np.log10(sig_pow / noise_pow + 1e-15)
    sfdr = 20 * np.log10(np.sqrt(sig_pow) /
                         np.max(np.delete(mag, sig_bin)) + 1e-15)
    return sndr, sfdr

sndr_raw, sfdr_raw = sndr_sfdr(code_bad)
sndr_cal, sfdr_cal = sndr_sfdr(code_cal)

print(json.dumps({
    "SNDR_raw(dB)": round(sndr_raw, 1),
    "SFDR_raw(dB)": round(sfdr_raw, 1),
    "SNDR_cal(dB)": round(sndr_cal, 1),
    "SFDR_cal(dB)": round(sfdr_cal, 1)
}, indent=2))

# ---------- 7. FFT plots ---------------------------
def plot_fft(code, fname, title):
    N = len(code)
    freq = np.fft.rfftfreq(N, d=1/FS) / 1e3    # kHz
    mag_db = 20 * np.log10(np.abs(
        np.fft.rfft((code - np.mean(code)) * np.hanning(N))) + 1e-12)

    plt.figure(figsize=(4.5, 2.2))
    plt.plot(freq, mag_db)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("dBFS")
    plt.title(title)
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

plot_fft(code_bad, "FFT_before.png", "FFT Before Calibration")
plot_fft(code_cal, "FFT_after.png",  "FFT After Calibration")
print("Saved  FFT_before.png  &  FFT_after.png")

