#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ========= 全域常數 =========
FS, FIN  = 500e6, 200e6          # 取樣率 / 輸入頻率
NBITS    = 13
VFS      = 1.0
NSAMPLES = 1 << 15               # 32 768 samples
MU       = 2e-8                  # LMS μ

# ========= Pipeline ADC ✧ 行為模型 =========
class Stage:
    def __init__(self, gain_err: float = 0.0):
        self.gain = 4 * (1 + gain_err)          # 理想增益 4 ± ε
    def process(self, v: np.ndarray):
        d  = np.clip(np.floor((v + 0.5) / 0.25), 0, 7)   # 3-bit sub-ADC
        vr = self.gain * (v - (d - 3.5) * 0.25)          # 殘值
        return d.astype(int), np.clip(vr, -1, 1)

class PipeADC:
    def __init__(self, gain_errs: list[float]):
        self.stages = [Stage(e) for e in gain_errs] + [Stage(0)]*(6-len(gain_errs))
    def convert(self, vin: np.ndarray):
        res, codes = vin.copy(), []
        for st in self.stages:
            d, res = st.process(res);  codes.append(d)
        # 串 18 bit → 擷取 MS 13 bit
        word = np.zeros_like(vin, dtype=int)
        for d in codes: word = (word << 3) | d
        return word >> (len(self.stages)*3 - NBITS), codes[:4]

# ========= 16-bit 參考 =========
ideal16 = lambda v: np.round(v / (VFS/(1<<16))).astype(int)

# ========= LMS =========
def lms_calibrate():
    t = np.arange(NSAMPLES)/FS
    vin = 0.5*VFS*np.sin(2*np.pi*FIN*t)

    adc = PipeADC([+0.05, -0.05, +0.04, -0.02])          # 前四級增益誤差
    _, codes = adc.convert(vin)
    ref = ideal16(vin)

    x = np.array([c-3 for c in codes]).astype(float)      # 4×N
    w = np.ones(4)
    err, w_hist = [], [w.copy()]

    for n in range(NSAMPLES):
        xn   = x[:, n]
        yhat = np.dot(w, xn)
        e    = (ref[n] >> 3) - yhat
        w   += 2*MU*e*xn
        err.append(e)

        # <<< 新增列印：每 2 k 次顯示即時 MSE >>>
        if n % 2000 == 0 and n:
            mse_now = np.mean(np.square(err[-2000:]))
            print(f"iter {n:5d}  |  local MSE = {mse_now:.4e}")

        if n % 200 == 0:
            w_hist.append(w.copy())

    return np.asarray(err), np.vstack(w_hist), w

# ========= 主程式 =========
if __name__ == "__main__":
    err, w_traj, w_final = lms_calibrate()

    # —— 最終統計 ——  (放在 show() 前保證終端能立即看到)
    mse_final = np.mean(np.square(err[-2000:]))
    print("\n=== result ===")
    print(f"final 2000 pints  MSE : {mse_final:.4e}")
    print("final w :", np.round(w_final, 5))

    # —— 畫圖 —— (阻塞在這之後)
    plt.figure()
    plt.semilogy(np.square(err))
    plt.xlabel("Iteration"); plt.ylabel("e²")
    plt.title("LMS error convergence")

    plt.figure()
    plt.plot(w_traj)
    plt.xlabel("snapshot index (every 200 samples)")
    plt.ylabel("weight value")
    plt.title("Weight evolution")
    plt.legend(["w₁","w₂","w₃","w₄"]); plt.grid(True)
    plt.show()
