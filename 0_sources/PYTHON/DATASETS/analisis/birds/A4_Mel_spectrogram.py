import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels, correlation2D, ss_2D
import numpy as np


def Mel_spectrogram(samples,
                n_fft=2000, hop_length= 100,
                magnitude = 'module',cmap='jet'):

    print('\nMel spectrogram:')
    print(' magnitude:', magnitude)
    print(' n_fft =',n_fft)
    print(' hop_length =',hop_length)

    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']
    components = [magnitude]

    data, ylabels = create_data_and_ylabels(sample_keys,states,components)
    c_data, c_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    ss_data, ss_ylabels = create_data_and_ylabels(sample_keys,states,components)


    ref_data = [[0],[0]]


    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):
            j+=1

            # np.array module to librosa stuff
            if magnitude == 'module' or magnitude == 'abs':
                wave, new_sr = arr2librosa(np.abs(sample.Data[j]),sr)
            elif magnitude == 'phase' or magnitude == 'angle':
                wave, new_sr = arr2librosa(np.angle(sample.Data[j]),sr)
            elif magnitude == 'real':
                wave, new_sr = arr2librosa(sample.Data[j].real,sr)
            elif magnitude == 'imag':
                wave, new_sr = arr2librosa(sample.Data[j].imag,sr)
            else:
                print('Invalid magnitude. Chose between: "module", "phase", "real" or "imag"')

            # Short-time Fourier transform (STFT)
            wave = np.abs(librosa.stft(wave, n_fft = n_fft, hop_length = hop_length))

            # Create the Mel Spectrograms
            wave = librosa.feature.melspectrogram(y = wave, sr=new_sr, n_mels=128)[0]

            # 2D correlation of the signals
            if i == 0:
                ref_data[j-1] = wave

            # Correlation between signals
            c_data[sample_key][state][magnitude] = [correlation2D(ref_data[j-1],wave,axis=1), new_sr]
            c_ylabels[sample_key][state] =  rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'

            # SS between signals
            ss = ss_2D(ref_data[j-1],wave,axis=1)
            z = np.linspace(sample.z[0],sample.z[-1],len(ss))
            ss_data[sample_key][state][magnitude] = [z, ss]
            ss_ylabels[sample_key][state] =  rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'


            # Amplitude to dB
            wave = librosa.amplitude_to_db(wave, ref = np.max)

            wave = wave.reshape(wave.shape[0],wave.shape[1]*wave.shape[2]) if len(wave.shape) == 3 else False

            # Store values
            data[sample_key][state][magnitude]    = [wave, new_sr]
            ylabels[sample_key][state] =  rf'$\nu \: [Hz]$'+'\n'+ rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'



    # Plot
    a_plot(data,ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'mel',dB = True)
    # Correlation plot
    a_plot(c_data,c_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'linear',dB = False)

    # SS plot
    a_plot(ss_data,ss_ylabels,hop_length = hop_length, type='')
    plt.show()
