import librosa
from scipy.io.wavfile import write
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def arr2librosa(array,sr=None):

    """ Function to create a .wav file from a numpy array and later open it with librosa

            : param array (np.array)  : a numpy array
            : param sr    (int)       : sample rate

            If array is compound any complex number

                : return wave (a_wave object): an object which contains

                                                    wave.real : the real part of the array
                                                    wave.imag : the imaginary part of the array
                                                    wave.mod  : the module of the array
                                                    wave.phase : the phase of the array

            else


                : return wave (librosa array): the original numpy array but in the librosa way

            : return new_sr (int) : because, for some reason is different from the original

            """

    if not sr:
        sr = len(array)

    if any(np.iscomplex(array)):

        class a_wave(object):
            def __init__(self):
                self.real   = ''
                self.imag   = ''
                self.mod    = ''
                self.phase  = ''

        wave = a_wave()

        write('temp.wav', sr, array.real)
        wave.real, new_sr  = librosa.load('temp.wav',sr = sr)
        os.remove('temp.wav')

        write('temp.wav', sr, array.imag)
        wave.imag, new_sr = librosa.load('temp.wav',sr = sr)
        os.remove('temp.wav')

        write('temp.wav', sr, np.abs(array))
        wave.mod, new_sr  = librosa.load('temp.wav',sr = sr)
        os.remove('temp.wav')

        write('temp.wav', sr, np.angle(array))
        wave.phase, new_sr = librosa.load('temp.wav',sr = sr)
        os.remove('temp.wav')

    else:

        write('temp.wav', sr, array)
        wave, new_sr = librosa.load('temp.wav')
        os.remove('temp.wav')



    return wave, new_sr

def correlation2D(Z1,Z2,axis=1):

    """ Function to compute correlation along some axis between two 2D arrays

        : param Z1 (2D array): First 2D array to compute correlation
        : param Z2 (2D array): Second 2D array to compute correlation

            Input arrays are assumed to be
                Z1[f,z] where f is the frequency and z is the lenght

        : optional axis (int): Axis to go

        : return corr2D (2D np.array): Array which contains correlation between arrays"""

    corr2D = list()
    Normalization = False
    for x in range(Z1.shape[axis]):

        if axis == 0:
            Y1 = Z1[x,:]
            Y2 = Z2[x,:]
        elif axis == 1:
            Y1 = Z1[:,x]
            Y2 = Z2[:,x]

        # Normalization
        if Normalization:
            Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
            Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

        # Cross corelation
        corr = np.correlate(Y1, Y2, mode='same')

        corr2D.append(corr)

    corr2D = np.array(corr2D)

    if axis == 1:
        corr2D = corr2D.T


    return corr2D

def ss_2D(Z1,Z2,axis=1):

    """ Function to spectral_shift along some axis between two 2D arrays

        : param Z1 (2D array): First 2D array to compute correlation
        : param Z2 (2D array): Second 2D array to compute correlation

        : optional axis (int): Axis to go

        : return corr2D (2D np.array): Array which contains correlation between arrays"""

    corr2D = correlation2D(Z1,Z2,axis=axis)

    ss_array = np.zeros_like(corr2D)

    arg_maxs = np.argmax(corr2D, axis=axis)

    return arg_maxs

def custom_stft(y,window=2000,delta=200,zchirp=False):

        """ Function to create a spectrogram based in the STFT algorithm
            but simplyfied

            : param y   (1D np.array or list)   : time series signal

            : optional window (int) : number of points taken to compute the FFT
            : optional delta  (int) : hop length between windows

            : return stft (2D np.array): spectrogram

        """

        if zchirp:
            from scipy.signal import czt, czt_points
            fs = 51101
            f1 = 0
            M = fs // 2  # Just positive frequencies, like rfft
            a = np.exp(-f1/fs)  # Starting point of the circle, radius < 1
            w = np.exp(-1j*np.pi/M)  # "Step size" of circle

        window = int(window/2)

        steps = range(window,len(y)-window+1,delta)
        stft  = list()
        for i in steps:

                yy = y[i-window:i+window]

                YY = czt(yy,M + 1, w, a)  if zchirp else np.fft.fft(yy)

                stft.append(YY)

        stft = np.array(stft)

        return stft.T


def create_data_and_ylabels(sample_keys,states,components):

    data = dict.fromkeys(sample_keys)
    for sample in data.keys():
        data[sample] = dict.fromkeys(states)
        for state in data[sample].keys():
            data[sample][state] = dict.fromkeys(components)

    import copy
    ylabels = copy.deepcopy(data)
    print('ylabels is data') if ylabels is data else False

    return data, ylabels

from .Za_plot import a_plot

from .Zb_plot import b_plot

from .Zc_plot import c_plot
