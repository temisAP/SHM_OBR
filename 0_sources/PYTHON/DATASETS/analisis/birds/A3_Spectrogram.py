import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels, correlation2D, ss_2D, custom_stft
import numpy as np
import scipy


def Spectrogram(samples,n_fft=2000, hop_length= 200,magnitude = 'module',cmap='jet'):

    print('\nSpectrogram:')
    print(' magnitude:', magnitude)
    print(' n_fft =',n_fft)
    print(' hop_length =',hop_length)

    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']
    components = [magnitude]

    data, ylabels       = create_data_and_ylabels(sample_keys,states,components)
    c_data, c_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    ss_data, ss_ylabels = create_data_and_ylabels(sample_keys,states,components)

    ref_data = [[0],[0]]
    base_corr = [[0],[0]]


    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):
            j+=1

            # Short-time Fourier transform (STFT)
            wave, new_sr = arr2librosa(sample.Data[j],sr)
            wave = (wave.real**2 + wave.imag**2)**0.5


            # np.array module to librosa stuff
            if magnitude == 'module' or magnitude == 'abs':
                wave = np.abs(librosa.stft(wave, n_fft = n_fft, hop_length = hop_length))
            elif magnitude == 'phase' or magnitude == 'angle':
                wave = np.angle(librosa.stft(wave, n_fft = n_fft, hop_length = hop_length))
            elif magnitude == 'real':
                wave = librosa.stft(wave, n_fft = n_fft, hop_length = hop_length)
                wave = wave.real
            elif magnitude == 'imag':
                wave = librosa.stft(wave, n_fft = n_fft, hop_length = hop_length)
                wave = wave.imag
            else:
                print('Invalid magnitude. Chose between: "module", "phase", "real" or "imag"')


            f,t,wave = scipy.signal.stft(sample.Data[j], fs=sr, nperseg=hop_length, noverlap=None, nfft=n_fft, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
            wave = np.abs(wave)

            wave = np.abs(custom_stft(sample.Data[j]))

            # Reference state
            if i == 0:
                ref_data[j-1] = wave
                base_corr[j-1] = correlation2D(ref_data[j-1],ref_data[j-1],axis=1)

            # Correlation between signals
            c_data[sample_key][state][magnitude] = [correlation2D(ref_data[j-1],wave,axis=1)-base_corr[j-1], new_sr]

            c_ylabels[sample_key][state] =  rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'

            # SS between signals
            ss = ss_2D(ref_data[j-1],wave,axis=1)
            z = np.linspace(sample.z[0],sample.z[-1],len(ss))
            ss_data[sample_key][state][magnitude] = [z, ss]
            ss_ylabels[sample_key][state] =  rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'


            # Amplitude to dB
            wave = librosa.amplitude_to_db(wave, ref = np.max)

            # Store values
            data[sample_key][state][magnitude]    = [wave, new_sr]
            ylabels[sample_key][state] =  rf'$\nu \: [Hz]$'+'\n'+ rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'



    # Plot
    a_plot(data,ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'log',dB = True)

    # Correlation plot
    a_plot(c_data,c_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'linear',dB = False)

    # SS plot
    a_plot(ss_data,ss_ylabels,hop_length = hop_length, type='')
    plt.show()
