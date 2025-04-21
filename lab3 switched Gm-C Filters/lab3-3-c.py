import numpy as np
import matplotlib.pyplot as plt

# --- 1. 參數 ---
C_R_list = [2e-12, 3e-12, 4e-12, 6e-12]  # 四顆不同大小的放電電容
C_sum    = sum(C_R_list) 

f_LO = 2.4e9
N    = 8
T_int = N / f_LO

# --- 2. 頻率軸 ---
f_min, f_max = 1e6, 1e10
num_points   = 2000
f = np.logspace(np.log10(f_min), np.log10(f_max), num_points)
w = 2 * np.pi * f

# --- 3. F(f) = [ (1 - e^{-jwT_int}) / (j w ) ] * [ (1 - e^{-jwT_int}) / ( j w T_int ) ]
Hw_num  = (1 - np.exp(-1j * w * T_int))
Ff      = (Hw_num / (1j * w)) * (Hw_num / (1j * w * T_int))

# --- 4. H_c(f) = [ 4 * F(f ) ] / sum_i CRi
#     (因為每顆對同一時間窗積分, 最後並聯讀出)
H_c = (len(C_R_list) * Ff) / C_sum

# --- 轉 dB ---
H_c_mag_dB = 20 * np.log10(np.abs(H_c))

plt.figure()
plt.semilogx(f, H_c_mag_dB, label='(c) 4 Discharged Capacitors, different sizes')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Case (c): Integrate-and-Dump, 4 Capacitors of Different Sizes')
plt.grid(True)
plt.legend()
plt.show()
