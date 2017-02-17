import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
import numpy as np

# TODO: finish this script
nf = 6
tr = 0.955
isi_max = 16
fc1 = 1/120
fc2 = 1/isi_max

fs = 1 / tr
w1 = 2 * fc1 / fs
w2 = 2 * fc2 / fs
#
N = 100000

x = np.random.rand(N)
f_step = fs/(len(x))
f_v = np.arange(1, (len(x)/2)+1)
f_v *= f_step

print("fc1 = {} ({})\nfc2 = {} ({})".format(fc1, w1, fc2, w2))
b, a = signal.iirfilter(nf, [w1, w2], rs=80, rp=0, btype='band', analog=False, ftype='butter')

y = signal.filtfilt(b, a, x)

xfft = fft(x)[:len(x)/2]
yfft = fft(y)[:len(x)/2]

exp_gain = 20 * np.log10(abs(yfft/xfft))

w, h = signal.freqz(b, a, N)
f_w = (w * fs) / 2
fig = plt.figure()
# plt.subplot(2, 1, 1)
# plt.semilogx(f_w, abs(h))#np.log10(abs(h)))
plt.semilogx(f_v, exp_gain)
plt.semilogx(f_v, 20*np.log10(abs(xfft)))
plt.semilogx(f_v, 20*np.log10(abs(yfft)))
plt.semilogx([fc1, fc1], [-30, 80], linewidth=2)
plt.semilogx([fc2, fc2], [-30, 80], linewidth=2)
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.legend(["x", "y", "fc1", "fc2"])
# ax.axis((0.0001, 10, -100, 40))
plt.grid(which='both', axis='both')
plt.title("fc1 = {}    fc2 = {}".format(fc1, fc2))

# plt.subplot(2, 1, 2)
# plt.plot(f_v, xfft)
# plt.plot(f_v, yfft)
# plt.legend(["In", "Out"])

plt.show()