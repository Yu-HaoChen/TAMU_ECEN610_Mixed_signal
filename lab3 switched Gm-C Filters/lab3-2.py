import numpy as np
import matplotlib.pyplot as plt

CH = 15.425e-12
CR = 0.5e-12
CS = CH + CR   # 15.925 pF
a1 = CH / CS   # 約 0.9686

f_LO = 2.4e9      # LO freq
N = 8
T_int = N / f_LO  # 积分區間
f = np.logspace(1, 11, 1000)  # 10 Hz ~ 100 GHz 
w = 2*np.pi*f

# Part 1: 連續積分(不放電) × a1
#         有限窗積分(會放電) × (1-a1)
# 注意整個電容名義上是 CS
Hw_part = a1 * (1/(1j*w)) + (1-a1)*( (1 - np.exp(-1j*w*T_int)) / (1j*w) )

# Part 2: 再乘上 ZOH
Hw_zoh = (1 - np.exp(-1j*w*T_int)) / (1j*w*T_int)

# 合併
H_new = (1/CS)*Hw_part * Hw_zoh

# Magnitude in dB
H_new_mag_dB = 20*np.log10(np.abs(H_new))

# Plot
plt.figure(figsize=(8,6))
plt.semilogx(f, H_new_mag_dB, label='New H(f) with CH & CR')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function with History Capacitor CH and Rotating Capacitor CR')
plt.grid(True)
plt.legend()
plt.show()
