import matplotlib.pyplot as plt
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels
import numpy as np
import sklearn

def Spectral_rolloff(samples,type='Real-Imaginary'):

    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']

    if type == 'Real-Imaginary':
        components = dict.fromkeys(['Real','Imaginary'])
    elif type == 'Module-Phase':
        components = dict.fromkeys(['Module','Phase'])
    else:
        print('Invalid type. Chose between: "Real-Imaginary" or "Module-Phase"')

    data, ylabels = create_data_and_ylabels(sample_keys,states,components)

    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):

            j+=1

            # np.array to librosa stuff
            wave, new_sr = arr2librosa(sample.Data[j],sr)

            if type == 'Real-Imaginary':
                components = {'Real': wave.real,'Imaginary': wave.imag}
            elif type == 'Module-Phase':
                components = {'Module': wave.mod,'Phase': wave.phase}
            else:
                print('Invalid type. Chose between: "Real-Imaginary" or "Module-Phase"')


            for label,component in components.items():

                # Spectral RollOff Vector
                spectral_rolloff = librosa.feature.spectral_rolloff( y = component, sr = new_sr )[0]

                # Computing the time variable for visualization
                frames = range(len(spectral_rolloff))

                # Converts frame counts to time (seconds)
                t = librosa.frames_to_time(frames)

                # Function that normalizes the Sound Data
                def normalize(x, axis=0):
                    return sklearn.preprocessing.minmax_scale(x, axis=axis)

                data[sample_key][state][label] = [t,normalize(spectral_rolloff)]
                ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'

    # Make the plot
    a_plot(data,ylabels,type='')

    plt.subplots_adjust(top=0.954,
                            bottom=0.091,
                            left=0.091,
                            right=0.985,
                            hspace=0.0,
                            wspace=0.0)

    plt.show()
