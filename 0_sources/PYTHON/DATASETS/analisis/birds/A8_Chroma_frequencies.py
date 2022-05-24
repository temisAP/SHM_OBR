import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels
import numpy as np


def Chroma_frequencies(samples,
                n_fft=2000, hop_length= 100,
                magnitude = 'module',cmap='jet'):

    print('\nChroma frequencies:')
    print(' magnitude:', magnitude)
    print(' n_fft =',n_fft)
    print(' hop_length =',hop_length)

    sample_keys = samples.keys()
    states = ['P','S']
    components = [magnitude]

    data, ylabels = create_data_and_ylabels(sample_keys,states,components)


    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):

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

            # Chromogram
            chromagram = librosa.feature.chroma_stft(y = wave, sr=new_sr, hop_length=hop_length)


            # Store values
            data[sample_key][state][magnitude]    = [chromagram, new_sr]
            ylabels[sample_key][state] =  rf'Pitch class'+'\n'+ rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'



    # Plot
    a_plot(data,ylabels,hop_length = hop_length, x_axis='time', y_axis='chroma',dB = False)
    plt.show()
