import matplotlib.pyplot as plt
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, create_data_and_ylabels, custom_stft
import numpy as np
from math import atan2
import sklearn


def Spectral_features(samples,
                        feature='Harmonics',magnitude='Module',
                        hop_length = 200, n_fft = 1000):

    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']
    components = [magnitude]


    data, ylabels       = create_data_and_ylabels(sample_keys,states,components)

    a_data, a_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    b_data, b_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    c_data, c_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    d_data, d_ylabels   = create_data_and_ylabels(sample_keys,states,components)

    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):

            if feature == 'Harmonics' or feature == 'Perceptual':

                # np.array to librosa stuff
                wave, new_sr = arr2librosa(sample.Data[j+1],sr)


                # Harmonics and perceptual
                if magnitude == 'Module' or magnitude == 'Phase':
                    wave.mod_harm, wave.mod_perc = librosa.effects.hpss(wave.mod)
                    wave.phase_harm, wave.phase_perc = librosa.effects.hpss(wave.phase)
                    wave_list = [[wave.mod_harm, wave.mod_perc],[wave.phase_harm, wave.phase_perc]]

                elif magnitude == 'Real' or magnitude == 'Imaginary':
                    wave.real_harm, wave.real_perc = librosa.effects.hpss(wave.real)
                    wave.imag_harm, wave.imag_perc = librosa.effects.hpss(wave.imag)
                    wave_list = [[wave.real_harm, wave.real_perc],[wave.imag_harm, wave.imag_perc]]

                # Save harmonics or perceptual
                for k,component in enumerate(components):
                    if feature == 'Harmonics':
                        data[sample_key][state][component] = [wave_list[k][0], new_sr]
                    elif feature == 'Perceptual':
                        data[sample_key][state][component] = [wave_list[k][1], new_sr]



            elif feature == 'Spectral centroid':

                if magnitude == 'Real':
                    S = custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length).real
                elif magnitude == 'Imaginary':
                    S = custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length).imag
                elif magnitude == 'Module':
                    S = np.abs(custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length))
                elif magnitude == 'Phase':
                    S = np.angle(custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length))
                else:
                    print('Invalid type. Chose between: "Real", "Imaginary", "Module" or "Phase"')
                    exit()

                new_sr = sr

                # Calculate the Spectral Centroids out of the spectrogram
                spectral_centroids = librosa.feature.spectral_centroid(S = S, sr=new_sr, hop_length=hop_length, n_fft = n_fft)[0]

                # Computing the time variable for visualization
                frames = range(len(spectral_centroids))

                # Converts frame counts to time (seconds)
                t = librosa.frames_to_time(frames)

                # Function that normalizes the Sound Data
                def normalize(x, axis=0):
                    return sklearn.preprocessing.minmax_scale(x, axis=axis)

                data[sample_key][state][magnitude] = [t,normalize(spectral_centroids)]
                ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'


            ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'

    # Make the plot
    a_plot(data,ylabels,type='',linewidth=1)

    # Comprarisson between signals
    c_plot(data,ylabels)


    plt.show()
