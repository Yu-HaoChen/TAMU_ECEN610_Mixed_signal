#!/usr/bin/env python3
# adc_linearity.py
# -----------------------------------------------------------
# 3‑bit ADC Transfer Curve, INL Calculation, and Plot
# -----------------------------------------------------------
#
# 已知:
#   DNL (in LSB) for codes 0 to 7:
#       0, -0.5, 0, +0.5, -1, +0.5, +0.5, 0
#
#   Offset error = +0.5 LSB
#   Full scale error = +0.5 LSB
#
# 本程式依據以下步驟計算：
# 1. 求出每一碼實際階寬： Δ_k = 1 + DNL_k (單位：LSB)
# 2. 從 offset 開始累加 Δ_k 取得每一碼的實際輸出值
# 3. 端點法：連接起點(0, offset)與末點(7, ideal_end)建立端點線
#    由於理想斜率為 1 LSB/code，所以端點線為： V_EP(k) = offset + k
# 4. INL 為每一碼實際輸出與端點線之差： INL(k) = V_actual(k) - V_EP(k)
#
# 最後印出各碼結果，並繪製轉換曲線及端點線圖

import numpy as np
import matplotlib.pyplot as plt

# ----------------- 0. 輸入數據 ------------------------------
# 3-bit ADC，共 8 個碼
dnl = np.array([0, -0.5, 0, +0.5, -1, +0.5, +0.5, 0.0])  # 每一碼的 DNL (單位：LSB)

offset_err = 0.5    # Offset error = +0.5 LSB
fs_err     = 0.5    # Full scale error = +0.5 LSB    (末碼輸出高於理想 0.5 LSB)

codes = np.arange(8)   # 輸入碼 0~7

# ----------------- 1. 計算實際階寬 ----------------------------
# 理想情況下每階寬為 1 LSB，加上 DNL 後每階寬變為：
step_width = 1 + dnl   # 每一步寬，單位 LSB

# ----------------- 2. 累加求得實際輸出 (V_actual) ---------------
# 以 offset 為起點，逐階累加步寬
v_actual = np.empty_like(step_width, dtype=float)
v_actual[0] = offset_err
for k in range(1, len(codes)):
    v_actual[k] = v_actual[k - 1] + step_width[k]

# ----------------- 3. 計算端點線 (Endpoint Line) --------------------
# 理想情況下，轉換曲線應從 (0, offset) 連到 (7, offset + 7)
# 故端點線斜率為 1 LSB/code，即： V_EP(k) = offset_err + k
v_ep = offset_err + codes

# ----------------- 4. 計算 INL --------------------
# INL(k) = 實際輸出電壓 V_actual(k) - 端點線 V_EP(k)
inl = v_actual - v_ep

# ----------------- 5. 列印結果 --------------------
print("3-bit ADC Linearity Analysis:")
print("---------------------------------------------------")
print(f"{'Code':>4} {'DNL (LSB)':>10} {'Actual V (LSB)':>15} {'INL (LSB)':>12}")
print("---------------------------------------------------")
for k in codes:
    print(f"{k:4d} {dnl[k]:10.2f} {v_actual[k]:15.2f} {inl[k]:12.2f}")

peak_inl = np.max(np.abs(inl))
print("\nPeak |INL| = {:.2f} LSB".format(peak_inl))

# ----------------- 6. 繪圖 --------------------
plt.figure(figsize=(6,5))

# 畫實際轉換曲線 (階梯狀圖)
plt.step(codes, v_actual, where='post', label='Actual transfer')

# 畫端點線 (理想線性轉換)
plt.plot(codes, v_ep, 'k--', label='Endpoint line')

plt.xlabel("Input Code")
plt.ylabel("Output (LSB)")
plt.title("3-bit ADC Transfer Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
