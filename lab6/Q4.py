
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# =========== Global constants ===========
FS, FIN  = 500e6, 200e6         # sample rate / input tone
NBITS    = 13
VFS      = 1.0
NSAMPLES = 1 << 15              # 32 768 samples
MU       = 0.5                  # NLMS step size  (0 < μ < 2)
EPS      = 1e-12                # stability guard
MSE_THR  = 1e-2                 # convergence threshold

# =========== Pipeline ADC behavioural model ===========
class Stage:
    """Single 2.5-bit MDAC with gain error only (for illustration)."""
    def __init__(self, gain_err: float = 0.0):
        self.gain = 4 * (1 + gain_err)          # ideal gain 4 ± ε
    def process(self, v: np.ndarray):
        d  = np.clip(np.floor((v + 0.5) / 0.25), 0, 7)   # 3-bit sub-ADC (Vref = 1 V)
        vr = self.gain * (v - (d - 3.5) * 0.25)          # residue
        return d.astype(int), np.clip(vr, -1, 1)

class PipeADC:
    """6-stage pipeline, returns 13-bit word and first-4-stage raw codes."""
    def __init__(self, gain_errs):
        self.stages = [Stage(e) for e in gain_errs] + [Stage(0.0)] * (6 - len(gain_errs))
    def convert(self, vin: np.ndarray):
        res, codes = vin.copy(), []
        for st in self.stages:
            d, res = st.process(res)
            codes.append(d)
        word = np.zeros_like(vin, dtype=int)
        for d in codes:                                   # concatenate 18 bits
            word = (word << 3) | d
        msb13 = word >> (len(self.stages) * 3 - NBITS)    # keep MS 13 bits
        return msb13, codes[:4]

# =========== Ideal 16-bit reference ===========
ideal16 = lambda v: np.round(v / (VFS / (1 << 16))).astype(int)

# =========== NLMS adaptation ===========
def nlms_calibrate():
    # stimulus: full-scale sine
    t   = np.arange(NSAMPLES) / FS
    vin = 0.5 * VFS * np.sin(2 * np.pi * FIN * t)

    # pipeline with gain errors in the first four stages
    adc = PipeADC([+0.05, -0.05, +0.04, -0.02])
    _, codes = adc.convert(vin)
    ref = ideal16(vin)

    # input vector for NLMS: centre codes (0-7) to (-3…+3)
    x = np.array([c - 3 for c in codes]).astype(float)    # shape 4 × N
    w = np.ones(4)                                        # initial weights
    err, w_hist = [], [w.copy()]
    conv_iter = None

    for n in range(NSAMPLES):
        xn   = x[:, n]
        power = np.dot(xn, xn) + EPS
        yhat  = np.dot(w, xn)
        e     = (ref[n] >> 3) - yhat                      # align to 13-bit range
        w    += 2 * MU * e * xn / power                  # NLMS update
        err.append(e)

        if n % 200 == 0:
            w_hist.append(w.copy())

        # convergence check
        if n > 2000 and conv_iter is None:
            mse_recent = np.mean(np.square(err[-2000:]))
            if mse_recent < MSE_THR:
                conv_iter = n

    return np.asarray(err), np.vstack(w_hist), w, conv_iter

# =========== Main ===========
if __name__ == "__main__":
    err, w_traj, w_final, conv_iter = nlms_calibrate()

    mse_final = np.mean(np.square(err[-2000:]))
    print("\n=== NLMS results ===")
    print(f"Final MSE over last 2k samples : {mse_final:.4e}")
    print("Final weights                 :", np.round(w_final, 5))
    if conv_iter is None:
        print(f"Did not reach MSE < {MSE_THR} within {NSAMPLES} iterations")
    else:
        print(f"Iterations to reach MSE < {MSE_THR} : {conv_iter}")

    # --- error convergence plot ---
    plt.figure()
    plt.semilogy(np.square(err))
    plt.xlabel("Iteration")
    plt.ylabel("e\u00b2")
    plt.title("NLMS error convergence")

    # --- weight evolution plot ---
    plt.figure()
    plt.plot(w_traj)
    plt.xlabel("Snapshot index  (every 200 samples)")
    plt.ylabel("Weight value")
    plt.title("NLMS weight evolution")
    plt.legend(["w₁", "w₂", "w₃", "w₄"], loc="best")
    plt.grid(True)
    plt.show()
