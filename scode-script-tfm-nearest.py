# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:47:42 2023

@author: xs16051
"""

# import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert
from scipy.io import loadmat

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.style': 'normal'})

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
    y = lfilter(b, a, data, axis)
    return y

#%% Load data

data = loadmat('sdata-5mhz-els18-steel5850.mat')
timx = data['exp_data']['time'][0,0][:,0]
time_data = data['exp_data']['time_data'][0,0]
tx = (data['exp_data']['tx'][0,0][0,:]-1).astype(int)
rx = (data['exp_data']['rx'][0,0][0,:]-1).astype(int)
el_xc = data['exp_data']['array'][0,0]['el_xc'][0,0][0,:].astype(float)
el_yc = data['exp_data']['array'][0,0]['el_yc'][0,0][0,:].astype(float)
el_zc = data['exp_data']['array'][0,0]['el_zc'][0,0][0,:].astype(float)
fs = 1 / (timx[2] - timx[1])
ph_velocity = 5850

#%% Apply butter bandpass filter

fc = 5e6 # centre frequency
fb = 50 # bandwidth of the bandpass filter
fo = 5   # butter filter order
lowcut = fc*(1-fb/200)  # Low cutoff frequency of the filter, Hz
highcut = fc*(1+fb/200)  # High cutoff frequency of the filter, Hz
data_flt = butter_bandpass_filter(time_data, lowcut, highcut, fs, fo)

# Perform Fourier transform
fft_pts = int(2**np.ceil(np.log2(time_data.shape[0])))
freq_spec = np.fft.fft(time_data, n=fft_pts, axis=0)
freq_spec = 2*freq_spec[0:int(fft_pts/2),:]
freq = np.linspace(0, (fs/fft_pts*(fft_pts/2-1)), int(fft_pts/2))
freq_flt = np.fft.fft(data_flt, n=fft_pts, axis=0)
freq_flt = 2*freq_flt[0:int(fft_pts/2),:]

#%% Plot the original and filtered signals

tr = 0
plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
plt.plot(timx*1e6, time_data[:, tr], label='Original signal')
plt.plot(timx*1e6, data_flt[:, tr], label='Filtered signal')
plt.grid(color='0.7', linestyle=':', linewidth=0.5)
plt.legend(frameon=False)
plt.xlabel('Time [$\mu$s]')
plt.ylabel('Amplitude')

plt.subplot(2,1,2)
plt.plot(freq*1e-6, abs(freq_spec[:, tr]), label='Original signal')
plt.plot(freq*1e-6, abs(freq_flt[:, tr]), label='Filtered signal')
plt.grid(color='0.7', linestyle=':', linewidth=0.5)
plt.xlim(0, 20)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%% TFM

tt = time.time()

x_size = 50 * 1e-3
z_size = 60 * 1e-3
p_size = 0.1 * 1e-3

x = np.linspace(-x_size/2, x_size/2, int(x_size/p_size)+1)
z = np.linspace(0, z_size, int(z_size/p_size)+1)
[x_mg, z_mg] = np.meshgrid(x, z)

delay = np.zeros([len(x_mg.flatten('F')), len(el_xc)])
for ii in range(0, len(el_xc), 1):
    delay[:, ii] = np.sqrt((x_mg.flatten('F') - el_xc[ii])**2 + z_mg.flatten('F')**2) / ph_velocity

II = np.zeros(x_mg.flatten('F').shape, dtype=complex)
for ii in range(0, time_data.shape[1], 1):
    idx = np.round((delay[:, tx[ii]] + delay[:, rx[ii]]) * fs)
    II += hilbert(data_flt[:, ii])[idx.astype(int)]

II = abs(II).reshape((len(x), len(z)))
II_db = 20 * np.log10(II/np.max(II))

print('TFM runs: ' + str(time.time()-tt) + ' seconds')

#%% Plot TFM

extent = [x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3]
dbscale = -40

plt.figure(figsize=(8, 6))
plt.imshow(II_db.T, extent=extent, vmin=dbscale, cmap='jet')
plt.grid(color='0.7', linestyle=':', linewidth=0.5)
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')
plt.colorbar(label='dB')
plt.tight_layout()
plt.show()

