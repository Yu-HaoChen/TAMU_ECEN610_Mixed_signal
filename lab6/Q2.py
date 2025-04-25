#!/usr/bin/env python3
# -------------------------------------------------------------
#  ELEN 610 – LAB 6  |  Task 2  Static-Error Sweep
# -------------------------------------------------------------
#  - Fs  = 500 MS/s   - Fin = 200 MHz   - 13-bit, 6-stage (2.5 b/stage)
#  - 掃描 6 種靜態誤差 → 找到造成 SNDR ≤ 10 dB 的臨界值
#  - 可載入簡易 Matlab-style 初始化檔以修改誤差範圍
# -------------------------------------------------------------
from __future__ import annotations
import re, argparse, pathlib, sys, numpy as np

# ---------- ADC / simulation constants ----------
FS       = 500e6          # sampling rate  (Hz)
FIN      = 200e6          # input tone     (Hz)
NBITS    = 13             # nominal bits
VFS      = 1.0            # full-scale (diff-peak, V)
NSAMPLES = 1 << 14        # FFT length

TARGET_SNDR = 10.0        # dB

# ---------- helper : SNR / SNDR ----------
def spectral_sndr(code: np.ndarray,
                  fs: float = FS,
                  fin: float = FIN) -> float:
    """Return SNDR (signal / noise+distortion) in dB."""
    code = code - np.mean(code)
    win  = np.blackman(len(code))
    spec = np.abs(np.fft.rfft(code * win)) ** 2
    freqs = np.fft.rfftfreq(len(code), 1 / fs)

    sig_bin   = np.argmin(np.abs(freqs - fin))
    signal_pw = spec[sig_bin]
    noise_pw  = spec.sum() - signal_pw
    return 10 * np.log10(signal_pw / noise_pw)

# ---------- simple behavioural ADC w/ error hooks ----------
class PipelineADC:
    """
    極簡 6-stage、2.5 bit/Stage Pipe-ADC，用於相對誤差掃描即可。
    • 每級 3 bit 直接串連再裁切至 13 bit
    • 只在「第一級」注入誤差 (對 SNDR 影響最明顯)
    """
    def __init__(self,
                 ota_gain      = np.inf,
                 ota_offset    = 0.0,
                 cap_mismatch  = 0.0,
                 cmp_offset    = 0.0,
                 op_bw         = np.inf,
                 nl_poly       = 0.0):
        self.A0   = ota_gain
        self.Vos  = ota_offset
        self.capm = cap_mismatch
        self.coff = cmp_offset
        self.bw   = op_bw
        self.nl3  = nl_poly

    # —— 2.5-bit MDAC building block ——
    @staticmethod
    def _sub_adc(v: np.ndarray, vref: float) -> np.ndarray:
        step = vref / 4.0
        return np.clip(np.floor((v + vref / 2) / step), 0, 7).astype(int)

    def convert(self, vin: np.ndarray) -> np.ndarray:
        res   = vin.copy()
        code  = np.zeros_like(vin, dtype=np.int64)

        # ------ inject errors only at first stage ------
        res  += self.coff                           # comparator offset
        res  *= 1 + self.capm                       # capacitor mismatch → gain error
        res  += self.Vos                            # OTA offset
        if np.isfinite(self.A0):                    # finite OTA gain
            res *= self.A0 / (1 + self.A0)
        res += self.nl3 * res**3                    # 3rd-order non-linearity

        # ------ ideal pipeline processing (6 × 2.5 b)  ------
        vref = VFS
        for _ in range(6):
            d   = self._sub_adc(res, vref)          # 3-bit coarse code
            res = 4 * (res - (d - 3.5) * (vref / 4))
            code = (code << 3) | d

        drop = 6 * 3 - NBITS
        return code >> drop

# ---------- default sweep ranges ----------
SWEEP_RANGES = {
    "ota_gain"     : np.logspace(5, 2, 40)[::-1],     # 100 k → 100
    "ota_offset"   : np.linspace(0, 200e-3, 40),      # 0 → 200 mV
    "cap_mismatch" : np.linspace(0, 5e-2, 40),        # 0 → 5 %
    "cmp_offset"   : np.linspace(0, 200e-3, 40),      # 0 → 200 mV
    "op_bw"        : np.logspace(9, 5, 40)[::-1],     # 1 GHz → 100 kHz
    "nl_poly"      : np.linspace(0, 0.5, 40),         # 0 → 50 % (3rd-order)
}

# ---------- optional : load Matlab-style init file ----------
def load_matlab_init(path: str) -> dict:
    """
    讀取極簡 .m / .txt :
    ota_gain    = 8e3;
    cap_mismatch= 0.02;
    (分號可有可無；支援科學記號)
    """
    init = {}
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+0-9.eE]+)")
    for line in pathlib.Path(path).read_text().splitlines():
        m = pattern.search(line)
        if m:
            k, v = m.groups()
            init[k] = float(v)
    return init

# ---------- main sweep ----------
def main():
    ap = argparse.ArgumentParser(description="Lab6 – Task2 static-error sweep")
    ap.add_argument("--init", help="Matlab init file : overrides sweep ranges / params")
    ap.add_argument("--plot", action="store_true", help="Show SNDR-vs-error plots")
    args = ap.parse_args()

    # sine stimulus
    t   = np.arange(NSAMPLES) / FS
    vin = 0.5 * VFS * np.sin(2 * np.pi * FIN * t)

    # allow init file to tweak ranges
    if args.init:
        user = load_matlab_init(args.init)
        for k in SWEEP_RANGES.keys():
            if k + "_range" in user:        # user can define custom range variable
                SWEEP_RANGES[k] = np.array(eval(user[k + "_range"]))

    import matplotlib.pyplot as plt
    thresholds = {}

    for name, sweep in SWEEP_RANGES.items():
        sndr_list = []
        for val in sweep:
            adc = PipelineADC(**{name: val})
            sndr = spectral_sndr(adc.convert(vin))
            sndr_list.append(sndr)
            if sndr <= TARGET_SNDR:
                thresholds[name] = val
                break
        else:
            thresholds[name] = np.nan   # never reached 10 dB

        # ------- optional plot -------
        if args.plot:
            x = sweep[: len(sndr_list)]
            plt.figure()
            plt.plot(x, sndr_list, marker="o")
            plt.axhline(TARGET_SNDR, color="r", linestyle="--")
            plt.xlabel(name)
            plt.ylabel("SNDR (dB)")
            plt.title(f"{name} sweep")
            plt.grid(True)

    # ------- summary -------
    print("\nThreshold where SNDR ≤ 10 dB")
    print("-" * 37)
    for k, v in thresholds.items():
        if np.isnan(v):
            print(f"{k:12s} :  > {SWEEP_RANGES[k][-1]}")
        else:
            print(f"{k:12s} :  {v}")

    if args.plot:
        plt.show()

if __name__ == "__main__":
    main()
