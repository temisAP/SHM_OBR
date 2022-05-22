import librosa
from scipy.io.wavfile import write
import os
import numpy as np

def arr2librosa(array,sr):

    if any(np.iscomplex(array)):

        class a_wave(object):
            def __init__(self):
                self.real = ''
                self.imag = ''

        wave = a_wave()

        write('temp.wav', sr, array.real)
        wave.real, new_sr  = librosa.load('temp.wav')
        os.remove('temp.wav')

        write('temp.wav', sr, array.imag)
        wave.imag, new_sr = librosa.load('temp.wav')
        os.remove('temp.wav')

    else:

        write('temp.wav', sr, array)
        wave, new_sr = librosa.load('temp.wav')
        os.remove('temp.wav')

    return wave, new_sr
