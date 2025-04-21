import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# ----------------------------------
# Parameters
# ----------------------------------
fs = 10e9                 # Sampling frequency (10 GHz)
T = 1 / fs                # Sampling period (100 ps)
T_charge = 50e-12         # Charging time (50 ps) within each sample period
tau = 12e-12              # RC time constant (12 ps)

N_bits = 7
V_fs = 1.0                # Full-scale voltage (1 V)
Delta = V_fs / (2**N_bits)  # Quantization step size

# Quantization noise variance (uniform distribution)
var_q = (Delta**2) / 12

# Number of samples for simulation
num_samples = 10000
t = np.arange(num_samples) * T

# ----------------------------------
# Generate multitone input signal (as in 2(b))
# Frequencies: 0.2, 0.58, 1, 1.7, and 2.4 GHz
# Each tone has an amplitude (set to 0.1 V here) and a random phase
frequencies = np.array([0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9])
phases = np.random.uniform(0, 2 * np.pi, size=frequencies.shape)
amplitude = 0.1

Vin = np.zeros(num_samples)
for f, phi in zip(frequencies, phases):
    Vin += amplitude * np.sin(2 * np.pi * f * t + phi)
# Scale signal to have a maximum amplitude of 0.5 V (as defined in the problem)
Vin = Vin / np.max(np.abs(Vin)) * 0.5

# ----------------------------------
# Simulate the RC sampling circuit
# We assume a sample-and-hold model:
# Each sampling period, the capacitor charges from its previous value toward the current Vin
V_RC = np.zeros(num_samples)
V_RC[0] = Vin[0]  # initial condition

# Calculate the charging factor for the exponential RC charging curve
alpha = 1 - np.exp(-T_charge / tau)
for n in range(1, num_samples):
    V_RC[n] = V_RC[n - 1] + alpha * (Vin[n] - V_RC[n - 1])

# ----------------------------------
# Define quantizer function (7-bit quantizer)
def quantize(x):
    # Round x to the nearest quantization level and clip between 0 and V_fs
    q = np.clip(np.round(x / Delta) * Delta, 0, V_fs)
    return q

# Ideal sampling: directly quantize the input signal
y_ideal = quantize(Vin)
# RC sampling followed by quantization
y_ADC = quantize(V_RC)

# Sampling error: difference between ADC output and ideal output
E = y_ADC - y_ideal
var_E = np.var(E)
ratio_a = var_E / var_q

print("=== 3(a) Sampling Error ===")
print(f"Quantization noise variance = {var_q:.3e} V^2")
print(f"Sampling error variance (E)  = {var_E:.3e} V^2")
print(f"Variance ratio (E/quantization noise) = {ratio_a:.3f}")

# ----------------------------------
# (b) FIR compensation to reduce the sampling error
# We will use an M-tap FIR filter (using M-1 previous ADC outputs) to estimate the sampling error.
Ms = np.arange(2, 11)  # FIR filter taps from 2 to 10
ratio_list = []       # List to store the ratio of error variance after compensation

# To avoid initial transients, ignore the first N_cut samples for estimation
N_cut = 1000

# Loop over different filter tap lengths M
for M in Ms:
    # Construct regression matrix X and target vector d for least squares estimation
    X = []
    d = []
    for n in range(N_cut, num_samples):
        if n - (M - 1) < 0:
            continue
        # Use M consecutive ADC outputs (from n-M+1 to n) as predictor features
        X.append(y_ADC[n - M + 1: n + 1])
        d.append(E[n])
    X = np.array(X)  # shape: (num_data, M)
    d = np.array(d)  # shape: (num_data,)
    
    # Least squares solution to estimate FIR coefficients h (d ≈ X * h)
    h, _, _, _ = lstsq(X, d, rcond=None)
    
    # Estimate the sampling error using the FIR filter over the whole sequence
    E_hat = np.zeros_like(E)
    for n in range(M - 1, num_samples):
        E_hat[n] = np.dot(y_ADC[n - M + 1: n + 1], h)
    
    # Compensated ADC output
    y_corr = y_ADC + E_hat
    E_corr = y_corr - y_ideal
    var_E_corr = np.var(E_corr[N_cut:])  # Avoid initial transients
    
    ratio_list.append(var_E_corr / var_q)
    print(f"M = {M}, compensated error variance ratio = {var_E_corr / var_q:.3f}")

# Plot the ratio of compensated error variance to quantization noise variance as M varies
plt.figure(figsize=(8, 5))
plt.plot(Ms, ratio_list, marker='o')
plt.xlabel("Number of FIR taps (M)")
plt.ylabel("Error variance / Quantization noise variance")
plt.title("FIR Compensation Effect: Error Variance Ratio vs. FIR Taps")
plt.grid(True)
plt.show()
