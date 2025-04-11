#!/usr/bin/env python3
# adc_hist_linearity.py
# -----------------------------------------------------------
# 4‑bit ADC ramp‑histogram linearity analysis (DNL / INL)
# (Version without pandas)
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -------- 0. 測試資料 -----------------------------------------------------
counts = np.array([
     43, 115,  85, 101, 122, 170,  75, 146,
    125,  60,  95,  95, 115,  40, 120, 242
], dtype=float)

n_bits  = 4
n_codes = 2 ** n_bits
codes   = np.arange(n_codes)

# -------- 1. 基本統計 -----------------------------------------------------
N_tot  = np.sum(counts)
N_avg  = N_tot / n_codes        # 理想每碼計數 (= 實際 1 LSB 寬度)

# -------- 2. DNL ----------------------------------------------------------
# DNL(k) = (N_k / N_avg) - 1
dnl = counts / N_avg - 1

# -------- 3. INL (endpoint 校正) -----------------------------------------
# 先做累積和: inl_raw[k] = sum(dnl[0]~dnl[k])
inl_raw = np.cumsum(dnl)

# 端點線補償：把 inl_raw 的首尾拉成 0
#   inl_raw[-1]: 最後一點的累積值
#   (inl_raw[-1]/(n_codes - 1)) * codes : 由 (0, inl_raw[-1]) 連成直線
inl = inl_raw - (inl_raw[-1] / (n_codes - 1)) * codes

# -------- 4. 峰值與單調判斷 ----------------------------------------------
peak_dnl = np.max(np.abs(dnl))
peak_inl = np.max(np.abs(inl))

# 單調 (monotonic) 判斷:
# 只要所有 DNL > -1 (LSB)，表示每階梯都不會倒退。
is_monotonic = np.all(dnl > -1)

# -------- 5. 列印結果 -----------------------------------------------------
print('\n=== 4‑bit ADC Histogram Linearity ===')
print(f'Total samples          : {int(N_tot)}')
print(f'Ideal counts / code    : {N_avg:.4f}\n')

# 打印表頭
print(f'{"Code":>5} {"Counts":>8} {"DNL (LSB)":>10} {"INL (LSB)":>10}')
print('-'*40)

for c in range(n_codes):
    print(f'{c:5d} {int(counts[c]):8d} {dnl[c]:10.3f} {inl[c]:10.3f}')

print('\nResults:')
print(f'Peak |DNL| = {peak_dnl:.3f} LSB')
print(f'Peak |INL| = {peak_inl:.3f} LSB')
print(f'Monotonic  : {"YES" if is_monotonic else "NO"}')

# -------- 6. (可選) 繪圖 --------------------------------------------------
plt.figure(figsize=(6,5))

plt.subplot(2,1,1)
plt.stem(codes, dnl, use_line_collection=True)
plt.title('Differential Non‑Linearity')
plt.xlabel('Code')
plt.ylabel('DNL (LSB)')
plt.grid(True)

plt.subplot(2,1,2)
plt.step(codes, inl, where='mid')
plt.title('Integral Non‑Linearity')
plt.xlabel('Code')
plt.ylabel('INL (LSB)')
plt.grid(True)

plt.tight_layout()
plt.show()
