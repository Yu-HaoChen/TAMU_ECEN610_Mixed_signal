import numpy as np
import matplotlib.pyplot as plt

C = 15.925e-12
f_LO = 2.4e9
N = 8
T_int = N / f_LO

# 針對 200 MHz ~ 400 MHz 之間做高解析度掃描 (線性分佈)
f_lin = np.linspace(2.0e8, 4.0e8, 20001)  # 2萬個點
w_lin = 2 * np.pi * f_lin

Hw_a_num = (1 - np.exp(-1j * w_lin * T_int))
Hw_a_int = Hw_a_num / (1j * w_lin)
Hw_a_hold = Hw_a_num / (1j * w_lin * T_int)
H_a = (1/C) * Hw_a_int * Hw_a_hold

H_a_mag_dB = 20 * np.log10(np.abs(H_a))

plt.figure()
plt.plot(f_lin, H_a_mag_dB, label='Case (a) around 300 MHz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Zoom in around 300 MHz (Case a)')
plt.grid(True)
plt.legend()
plt.show()