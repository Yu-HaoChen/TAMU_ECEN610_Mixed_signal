import numpy as np
import matplotlib.pyplot as plt

#-----------------------------
# 參數設定
#-----------------------------
f_in = 1e9       # 輸入正弦波頻率 (1 GHz)
f_s  = 10e9      # 取樣頻率 (10 GHz)
tau  = 10e-12    # RC 時常數 (10 ps)

T_in = 1 / f_in  # 輸入正弦波週期
T_s  = 1 / f_s   # 取樣週期 (100 ps)

# 模擬總時間 (這裡取輸入信號的 5 週期為例)
t_end = 5 * T_in

# 設定數值積分的時間步距 (比取樣週期更細，避免數值誤差)
dt = T_s / 50
t = np.arange(0, t_end, dt)

#-----------------------------
# 準備儲存結果的陣列
#-----------------------------
Vin  = np.sin(2 * np.pi * f_in * t)  # 輸入正弦波
Vout = np.zeros_like(t)              # 輸出波形(電容電壓)
Vout[0] = 0.0                        # 假設一開始電容電壓為 0

#-----------------------------
# 幫助函式: 判斷開關在某個時間點是否導通
# 這裡簡單假設每個取樣週期的前半段(50%) ON，後半段 OFF
#-----------------------------
def is_switch_on(time):
    # 先求出該時刻落在第幾個取樣週期
    n = int(time // T_s)
    # 取樣週期起始點
    t_period_start = n * T_s
    # 當前時刻在此取樣週期的相對時間
    t_rel = time - t_period_start
    
    # 若相對時間 < T_s/2，則視為開關 ON，否則 OFF
    if t_rel < (T_s / 2):
        return True
    else:
        return False

#-----------------------------
# 主迴圈: 使用 Euler method 進行數值模擬
#-----------------------------
for i in range(len(t) - 1):
    # 取得當前時刻的輸入電壓
    vin_now = Vin[i]
    vout_now = Vout[i]
    
    # 判斷開關狀態
    if is_switch_on(t[i]):
        # 開關 ON: dVout/dt = (Vin - Vout)/tau
        dvout_dt = (vin_now - vout_now) / tau
        Vout[i+1] = vout_now + dvout_dt * dt
    else:
        # 開關 OFF: 電容電壓保持不變
        Vout[i+1] = vout_now

#-----------------------------
# 繪圖
#-----------------------------
plt.figure(figsize=(8, 5))
plt.plot(t*1e9, Vin, label='Vin (1 GHz)', linewidth=1.5)
plt.plot(t*1e9, Vout, label='Vout (sampled)', linewidth=2)
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.title('Sampling Circuit Output (1 GHz input, 10 GHz sampling, tau=10 ps)')
plt.legend()
plt.grid(True)
plt.show()
