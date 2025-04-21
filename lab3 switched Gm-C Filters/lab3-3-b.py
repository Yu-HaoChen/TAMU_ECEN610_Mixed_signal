import numpy as np
import matplotlib.pyplot as plt

# --- 參數設定 ---
C_R  = 1e-12
f_LO = 2.4e9
N    = 8
T_int = N / f_LO

f_min, f_max = 1e6, 1e10
f = np.logspace(np.log10(f_min), np.log10(f_max), 2000)
w = 2 * np.pi * f

# (b) Continuous Integration
Hw_b_int  = 1 / (1j * w)                   # 1/(j w)
Hw_b_hold = (1 - np.exp(-1j * w * T_int)) / (1j * w * T_int)
H_b       = (1 / C_R) * Hw_b_int * Hw_b_hold

H_b_mag_dB = 20*np.log10(np.abs(H_b))

plt.figure()
plt.semilogx(f, H_b_mag_dB, label='(b) 4 Caps Never Discharged')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Case (b): Continuous Integration')
plt.grid(True)
plt.legend()
plt.show()
