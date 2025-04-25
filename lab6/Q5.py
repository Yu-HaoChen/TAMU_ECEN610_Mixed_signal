#!/usr/bin/env python3
# -------------------------------------------------------------
#  ELEN 610 – LAB 6 | Task 5  High-order NLMS with Noise & Finite GBW
# -------------------------------------------------------------
#  • Adds white noise so input SNR ≈ 80 dB            (Fin = 200 MHz)
#  • Op-Amp GBW = 0.8 × value required for 13 bit     (Fs = 500 MS/s)
#  • Open-loop gain includes 2ⁿ–5ⁿ-order terms:
#        k₂ = 0.10 A₀ ,  k₃ = 0.20 A₀ ,
#        k₄ = 0.15 A₀ ,  k₅ = 0.10 A₀
#  • NLMS basis vector: [x, x², x³, x⁴, x⁵] (up to chosen order)
#  • Runs for decimation factors n = 10, 100, 1000, 10000
#    → prints iterations needed to reach MSE < 1e-2
# -------------------------------------------------------------
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ---------- constants ----------
FS, FIN  = 500e6, 200e6
NBITS    = 13
VFS      = 1.0
NS       = 1 << 17                 # 131 072 samples
MU       = 0.6                     # NLMS μ   (0<μ<2)
EPS      = 1e-12                   # avoid /0
MSE_THR  = 1e-2                    # convergence criterion
DECIMS   = [10, 100, 1000, 10000]  # decimation factors to test

# ---------- helper: add noise for 80 dB SNR ----------
def sine_plus_noise(fin=FIN, fs=FS, nsamp=NS, snr_db=80):
    t   = np.arange(nsamp) / fs
    sig = 0.5 * VFS * np.sin(2*np.pi*fin*t)
    sig_rms = np.sqrt(np.mean(sig**2))
    noise_rms = sig_rms / 10**(snr_db/20)
    sig += np.random.normal(scale=noise_rms, size=nsamp)
    return sig

# ---------- very-light pipeline ADC w/ GBW + non-linearities ----------
class SimpleADC:
    def __init__(self, gbw_factor: float = 0.8,
                 k2=0.10, k3=0.20, k4=0.15, k5=0.10):
        self.alpha = gbw_factor / (1 + gbw_factor)   # first-order settling loss
        self.k2, self.k3, self.k4, self.k5 = k2, k3, k4, k5
        self.q_step = VFS / (1 << NBITS)

    def convert(self, vin: np.ndarray) -> np.ndarray:
        # linear path + finite-GBW attenuation
        v = self.alpha * vin
        # polynomial open-loop distortion (relative to A0)
        v += self.k2*v**2 + self.k3*v**3 + self.k4*v**4 + self.k5*v**5
        # ideal 13-bit quantizer
        return np.round(v / self.q_step).astype(int)

# ---------- ideal 16-bit reference ----------
ref16 = lambda v: np.round(v / (VFS / (1<<16))).astype(int)

# ---------- NLMS routine (poly order selectable) ----------
def run_nlms(order: int, decim: int):
    vin   = sine_plus_noise()
    adc   = SimpleADC()
    meas  = adc.convert(vin)
    ref   = ref16(vin) >> 3                  # align to 13-bit range

    # build basis [x, x², …, x^order]  (x = scaled measurement)
    x1 = meas.astype(float)
    x  = np.vstack([x1**n for n in range(1, order+1)])   # shape (order, N)
    w  = np.ones(order)
    err = []
    for n in range(0, NS, decim):          # decimated update
        xn = x[:, n]
        p  = np.dot(xn, xn) + EPS
        e  = ref[n] - np.dot(w, xn)
        w += 2*MU*e*xn/p
        err.append(e)
        if n > 2000 and np.mean(np.square(err[-2000//decim:])) < MSE_THR:
            return n, err, w               # iterations to converge
    return None, err, w                    # not converged

# ---------- sweep decimation factors ----------
if __name__ == "__main__":
    order = 5
    print(f"\n=== Task 5 : NLMS up to order {order} ===")
    for n in DECIMS:
        it, err, w = run_nlms(order, n)
        if it is None:
            print(f"decim={n:5d}  →  not converged within {NS} samples")
        else:
            print(f"decim={n:5d}  →  converged in {it:6d} iterations "
                  f"(≈ {it/FS*1e6:.1f} µs)")

    # optional: show last run’s convergence curve
    plt.semilogy(np.square(err))
    plt.title(f"NLMS error (decim={DECIMS[-1]})")
    plt.xlabel("update index"); plt.ylabel("e\u00b2")
    plt.grid(True)
    plt.show()
