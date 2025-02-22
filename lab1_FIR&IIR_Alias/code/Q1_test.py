import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# === random signal ===
np.random.seed(42)  # 設定隨機種子
N = 500  # 訊號長度
x = np.random.randn(N)  # 隨機白雜訊

# === FIR stable ===
fir_b = [1, 1, 1, 1, 1]  # FIR numerator coefficients
fir_a = [1]  # FIR denominator
y_fir = signal.lfilter(fir_b, fir_a, x)

# === IIR stable ===
iir_stable_b = [1, 1]
iir_stable_a = [1, -0.9]  # Pole at 0.9 (inside unit circle)
y_iir_stable = signal.lfilter(iir_stable_b, iir_stable_a, x)

# === IIR unstable ===
iir_unstable_b = [1, 1]
iir_unstable_a = [1, 1.2]  # Pole at 1.2 (outside unit circle)
y_iir_unstable = signal.lfilter(iir_unstable_b, iir_unstable_a, x)

# === 繪製圖形 ===
fig, axes = plt.subplots(4, 1, figsize=(10, 12))

# 原始輸入訊號
axes[0].plot(x, color='black')
axes[0].set_title("Input Signal to Filter (Random Noise)", loc="left")
axes[0].set_xlabel("Sample Index")
axes[0].set_ylabel("Amplitude")

# FIR 濾波結果
axes[1].plot(y_fir, color='blue')
axes[1].set_title("Output from FIR Filter (Stable)", loc="left")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Amplitude")

# IIR 穩定濾波結果
axes[2].plot(y_iir_stable, color='green')
axes[2].set_title("Output from Stable IIR Filter", loc="left")
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("Amplitude")

# IIR 不穩定濾波結果
axes[3].plot(y_iir_unstable, color='red')
axes[3].set_title("Output from Unstable IIR Filter (Poles Outside Circle)", loc="left")
axes[3].set_xlabel("Sample Index")
axes[3].set_ylabel("Amplitude")

# 調整間距
plt.tight_layout()
plt.show()
