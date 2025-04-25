#!/usr/bin/env python3
"""
Lab 6 – Task 1  :  Ideal 13-bit ADC  •  SNR calculation
-------------------------------------------------------
• Fs  = 500 MS/s   • Fin = 200 MHz   • VFS = 1 V (diff-peak)
• Full-scale sine → ideal 13-bit quantiser → FFT → SNR
"""
import numpy as np

# ----- global constants -----
FS   = 500e6       # sample rate  (Hz)
FIN  = 200e6       # input tone   (Hz)
NB   = 13          # ADC bits
VFS  = 1.0         # differential-peak full scale (V)
N    = 1 << 14     # FFT length (power-of-two)

# ----- stimulus -----
t   = np.arange(N) / FS
vin = 0.5 * VFS * np.sin(2 * np.pi * FIN * t)   # –1 LSB head-room

# ----- ideal 13-bit quantiser -----
q_step = VFS / (2 ** NB)            # LSB size
codes  = np.round(vin / q_step)     # integer output (two's-complement)
vout   = codes * q_step             # DAC-reconstructed value
q_err  = vin - vout                 # quantisation error

# ----- SNR -----
signal_rms = np.sqrt(np.mean(vin   ** 2))
noise_rms  = np.sqrt(np.mean(q_err ** 2))
snr_db     = 20 * np.log10(signal_rms / noise_rms)

print(f"SNR = {snr_db:.2f} dB   (quantisation limit ≈ {6.02*NB + 1.76:.2f} dB)")


