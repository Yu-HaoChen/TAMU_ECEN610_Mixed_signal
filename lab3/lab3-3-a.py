import numpy as np
import matplotlib.pyplot as plt

# --- 參數設定 ---
C_R  = 1e-12         # 單顆電容值 (假設四顆相同)
f_LO = 2.4e9         # LO 頻率
N    = 8
T_int = N / f_LO     # 積分時間

# --- 頻率軸 ---
f_min, f_max = 1e6, 1e10   # 1 MHz ~ 10 GHz
num_points   = 2000
f = np.logspace(np.log10(f_min), np.log10(f_max), num_points)
w = 2 * np.pi * f

# --- (a) Integrate & Dump 傳遞函式 ---
Hw_num   = (1 - np.exp(-1j * w * T_int))        # 分子 1 - e^{-j w T_int}
Hw_int   = Hw_num / (1j * w)                    # 有限窗積分
Hw_hold  = Hw_num / (1j * w * T_int)            # 零階保持(ZOH)
H_a      = (1 / C_R) * Hw_int * Hw_hold         # 乘上 1/C

# --- 轉成 dB ---
H_a_mag_dB = 20*np.log10(np.abs(H_a))

plt.figure()
plt.semilogx(f, H_a_mag_dB, label='(a) 4 Caps Discharged')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Case (a): Integrate-and-Dump')
plt.grid(True)
plt.legend()
plt.show()
