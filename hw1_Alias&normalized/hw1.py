import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

Fs = float(input("Fs (Hz): "))
Fin = float(input("Fin (Hz): "))
N = int(input("FFT discrete point N: "))  # EX: N=100 fin(Hz) cut into x[1-100] discrete point x[y]= fin/N * y
M = int(input("Decimation Factor M: "))  # Decimation Factor EX: Fs/M into FsD

# After decoimate FsD and ND
FsD = Fs / M
ND = N // M  # 降頻後的 FFT 點數

# Fa (original)
k = round(Fin / Fs) # closest kFs
Fa = abs(Fin - k * Fs) # Frequnece alias back to Nyquist BW

# FaD (mod to insure between FsD)
FaD = Fa % FsD

# bin size
bin_size = Fs / N
bin_size_D = FsD / ND

# Fix alias freq，fit FFT bin
normalized_Fa = round(Fa / bin_size) * bin_size
normalized_FaD = round(FaD/ bin_size_D) * bin_size_D

#-----------------------------------------------------------------------------------------------------------------

# time
t = np.arange(N) / Fs  # original time
t_decimated = np.arange(ND) / FsD  # D time

# Orignial Fin
x_original = np.sin(2 * np.pi * Fin * t)
x_alias = np.sin(2 * np.pi * normalized_Fa * t)  # 修正後 alias 頻率

# decimated freq
x_decimated = x_original[::M] # FsD= Fs/M

# FFT [:N//2]slicing
X_original = np.abs(fft(x_original))  # fft abs conjurgrate for Magnitude not Phase
X_decimated = np.abs(fft(x_decimated))  # fft output [-Fs/2- Fs/2] only keep the 0- Fs/2

# fft (0, Fs/N, Fs/2N ... {N/(2-1)Fs}/N, -Fs/2 )
# [:N//2] for the first half ( to show nyquist BW)
# [N//2:] for the second half ( negative part )

freqs = np.fft.fftfreq(N, d=1/Fs)   #  FFT freq
freqs_decimated = np.fft.fftfreq(ND, d=1/FsD)  # D FFT freq

#--------------------------------------------------------------------------------------------------------------------

# Print
print("\noutcome:")
print(f"Fs: {Fs} Hz, FsD (decimated) = {FsD} Hz")
print(f"Fin: {Fin} Hz")
print(f"k : {k}")
print(f"Fa (original): {Fa} Hz,  normalized Fa = {normalized_Fa} Hz")
print(f"FaD = {FaD} Hz, normalized FaD = {normalized_FaD} Hz")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Upper left: original signal (time domain)
axes[0, 0].plot(t[:200], x_original[:200], color='blue')
axes[0, 0].set_title(f"Time Domain (Fin = {Fin} Hz)")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")

# Upper right: signal after downconversion (time domain)
axes[0, 1].plot(t_decimated[:200], x_decimated[:200], color='red')
axes[0, 1].set_title(f"Decimated Signal (FsD = {FsD} Hz)")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Amplitude")

# Bottom left: original FFT spectrum
#axes[1, 0].plot(freqs, X_original, label=f"Fin = {Fin} Hz", color='blue')
axes[1, 0].axvline(Fin, color='blue', label=f"Fin = {Fin} Hz")  # 原始頻率標示
axes[1, 0].axvline(Fa, color='green', linestyle=":", label=f"Fa = {Fa} Hz")
axes[1, 0].axvline(normalized_Fa, color='red', linestyle=":", label=f"Fa normalized = {normalized_Fa} Hz")  # Alias 頻率標示
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("FFT Magnitude")
axes[1, 0].set_title("Original FFT Spectrum")
axes[1, 0].legend()
axes[1, 0].grid()

# Bottom right: FFT spectrum after downconversion
#axes[1, 1].plot(freqs_decimated, X_decimated, label=f"Fa' = {normalized_FaD} Hz", color='red')
axes[1, 1].axvline(Fa, color='green', linestyle=":", label=f"FaD = {FaD} Hz")
axes[1, 1].axvline(normalized_FaD, color='red',  label=f"FaD normalized = {normalized_FaD} Hz")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("FFT Magnitude")
axes[1, 1].set_title("Decimated FFT Spectrum")
axes[1, 1].legend()
axes[1, 1].grid()

fig.suptitle("Original and Decimated Signals", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

