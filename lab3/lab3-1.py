import numpy as np
import matplotlib.pyplot as plt

#=== 1. 參數設定 ===#
C = 15.925e-12       # 電容值 (F)
f_LO = 2.4e9         # LO 頻率 (Hz)
N = 8
T_int = N / f_LO     # 積分(取樣)時間 (s)

#=== 2. 建立頻率軸 (10 Hz ~ 4.8 GHz) ===#
f_min = 10
f_max = 4.8e9
num_points = 1000    # 頻率掃描點數
f = np.logspace(np.log10(f_min), np.log10(f_max), num_points)
w = 2 * np.pi * f

#=== 3. (a) 每次讀出後放電 (Integrate-and-Dump) ===#
# H_a(f) = (1/C)*[(1 - e^{-j w T_int}) / (j w)] * [(1 - e^{-j w T_int}) / (j w T_int)]
Hw_a_num = (1 - np.exp(-1j * w * T_int))     # 分子 1 - e^{-j w T_int}
Hw_a_int = Hw_a_num / (1j * w)              # 有限積分 (在 T_int 內)
Hw_a_hold = Hw_a_num / (1j * w * T_int)     # 零階保持 (ZOH)
H_a = (1/C) * Hw_a_int * Hw_a_hold

#=== 4. (b) 從不放電 (Continuous Integration) ===#
# H_b(f) = (1/C)*[1/(j w)] * [(1 - e^{-j w T_int})/(j w T_int)]
Hw_b_int = 1 / (1j * w)                     # 連續積分 1/(j w)
Hw_b_hold = Hw_a_num / (1j * w * T_int)     # 零階保持 (ZOH)
H_b = (1/C) * Hw_b_int * Hw_b_hold

#=== 5. 轉成 dB ===#
H_a_mag_dB = 20 * np.log10(np.abs(H_a))
H_b_mag_dB = 20 * np.log10(np.abs(H_b))

#=== 6. 繪圖 ===#
plt.figure(figsize=(8,6))
plt.semilogx(f, H_a_mag_dB, label='Case (a) discharge')
plt.semilogx(f, H_b_mag_dB, label='Case (b) integrate')

plt.xlim(1e2, 4.8e9)
#plt.ylim([-60, 40])  
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('H(f) = Vo(f) / I_RF(f)')
plt.grid(True)
plt.legend()
plt.show()

