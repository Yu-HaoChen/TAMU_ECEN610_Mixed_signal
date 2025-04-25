#!/usr/bin/env python3
# -------------------------------------------------------------
#  ELEN 610 – LAB 6 | Task 7  128-Tone BPSK → ADC → NLMS → DFT
# -------------------------------------------------------------
#  • Fs = 500 MS/s                                  (fixed)
#  • 128 BPSK-modulated tones occupy 0 – 200 MHz
#        tone spacing  Δf = 200 MHz / 128 = 1.5625 MHz
#        ⇒ choose DFT length  L = Fs / Δf = 500 / 1.5625 = 320
#  • Perfect sync assumed → DFT bin k = m (m = 0…127)
#  • Four decimation factors n = {10, 100, 1000, 10000}
#  • Case A : linear errors only  (finite A0, offset, cap-mismatch, cmp-offset)
#    Case B : add 2ⁿ–5ⁿ-order non-linearities
#  • NLMS up to 5th-order, convergence tested per n
#  • After convergence run DFT, compute symbol MSE & BER
# -------------------------------------------------------------
from __future__ import annotations
import numpy as np

# ---------- constants ----------
FS, BW, NTONE = 500e6, 200e6, 128
DTF_LEN       = 320                         # exact bin spacing 1.5625 MHz
NBITS, VFS    = 13, 1.0
MU, EPS       = 0.6, 1e-12
DECIMS        = [10, 100, 1000, 10000]
MSE_THR       = 1e-3                        # convergence threshold
RAND          = np.random.default_rng(0xADC)

# ---------- multitone BPSK generator ----------
def multitone_bpsk(nsamp: int):
    sym = RAND.choice([-1.0, 1.0], NTONE)          # BPSK symbols
    amps = sym / np.sqrt(NTONE)                    # power normalise
    t = np.arange(nsamp) / FS
    freqs = np.arange(NTONE) * (BW / NTONE)        # 0 … 198.4375 MHz
    signal = 0.5*VFS * np.sum(amps[:, None] * np.cos(2*np.pi*freqs[:, None]*t), axis=0)
    return signal, sym

# ---------- ADC model (linear + optional non-linear) ----------
class ErrorADC:
    def __init__(self, linear_only=True):
        self.A0   = 2e3           # finite open-loop gain
        self.gbw  = 0.8           # 80 % GBW factor → first-order attenuation
        self.off  = 10e-3         # OTA offset 10 mV
        self.cmis = 0.01          # 1 % capacitor mismatch
        self.cmp  = 2e-3          # comparator offset 2 mV
        self.poly = (0.10, 0.20, 0.15, 0.10) if not linear_only else (0,0,0,0)

    def convert(self, x: np.ndarray) -> np.ndarray:
        v  = x + self.cmp
        v *= (1 + self.cmis)
        v += self.off
        v *= self.A0 / (1 + self.A0)               # finite A0
        v *= self.gbw / (1 + self.gbw)             # 1-p settling
        # polynomial non-linearities
        k2,k3,k4,k5 = self.poly
        v += k2*v**2 + k3*v**3 + k4*v**4 + k5*v**5
        step = VFS / (1<<NBITS)
        return np.round(v/step).astype(int)

# ---------- ideal 16-bit reference ----------
ideal16 = lambda v: np.round(v / (VFS / (1<<16))).astype(int) >> 3  # align to 13-bit

# ---------- NLMS with polynomial basis (up to 5th) ----------
def nlms_correct(meas, ref, order, decim):
    N = len(meas)
    x = np.vstack([meas.astype(float)**p for p in range(1, order+1)])  # order×N
    w = np.ones(order)
    err=[]
    for n in range(0, N, decim):
        xn = x[:, n]
        p  = np.dot(xn,xn) + EPS
        e  = ref[n] - np.dot(w,xn)
        w += 2*MU*e*xn/p
        err.append(e)
        if n>10_000 and np.mean(np.square(err[-5_000:])) < MSE_THR:
            break
    return w, n, np.asarray(err)

# ---------- DFT demod + BER ----------
def ber_and_mse(codes, sym_true):
    v = codes.astype(float)
    bins = np.fft.fft(v, n=DTF_LEN) / DTF_LEN      # DFT
    sym_est = np.sign(bins[:NTONE].real)           # simple slicer
    ber  = np.mean(sym_est != sym_true)
    mse  = np.mean((bins[:NTONE].real - sym_true)**2)
    return mse, ber

# ---------- run one scenario ----------
def run(decim, linear_only):
    nsamp = 10*DTF_LEN * decim                     # guarantee many updates
    sig, sym = multitone_bpsk(nsamp)
    adc  = ErrorADC(linear_only)
    meas = adc.convert(sig)
    ref  = ideal16(sig)

    # NLMS order = 1 for linear-only, 5 otherwise
    order = 1 if linear_only else 5
    w, its, err = nlms_correct(meas, ref, order, decim)

    # apply static correction
    xpoly = sum(w[i-1]*meas**i for i in range(1, order+1))
    mse, ber = ber_and_mse(xpoly, sym)
    return its, mse, ber

# ---------- main sweep ----------
if __name__ == "__main__":
    for case in ("Linear-only", "Linear+Non-linear"):
        lin_only = (case=="Linear-only")
        print(f"\n=== {case} errors ===")
        for n in DECIMS:
            its, mse, ber = run(n, lin_only)
            print(f"decim={n:5d} | iter to conv ≈ {its:6d}"
                  f" | MSE={mse:.3e} | BER={ber:.3e}")
