import matplotlib.pyplot as plt
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, c_plot, create_data_and_ylabels, custom_stft, correlation2D
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import check_memory


def Features(samples,
                feature='Harmonics',magnitude='Module',orders=[1,2,3,20,50], czt = True,
                hop_length = 200, n_fft = 1000):

    print(f'\n{feature}:')
    print(' magnitude:', magnitude)
    print(' n_fft =',n_fft)
    print(' hop_length =',hop_length)

    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']
    components = ['Spectrogram','Cross correlated spectrogram','Comparisson of cross correlated spectrograms','Autocorrelated spectrogram','Comparisson of autocorrelated spectrograms']

    data, ylabels       = create_data_and_ylabels(sample_keys,states,components)

    c_data, c_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    d_data, d_ylabels   = create_data_and_ylabels(sample_keys,states,components)

    t_data, t_ylabels       = create_data_and_ylabels(sample_keys,states,components)
    oenv_data, oenv_ylabels = create_data_and_ylabels(sample_keys,states,components)

    ref_S       = [[0],[0],[0]]
    base_corr   = [[0],[0],[0]]

    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):

            # Labels
            ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'


            if feature == 'Harmonics' or feature == 'Percusive':

                components = [magnitude]

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
                    elif feature == 'Percusive':
                        data[sample_key][state][component] = [wave_list[k][1], new_sr]

                type = 'librosa'





            elif 'Spectral' in feature or 'RMS' in feature:

                if czt:
                    from scipy.signal import czt, czt_points
                    fs = sr
                    f1 = 0
                    M = fs // 2  # Just positive frequencies, like rfft
                    a = np.exp(-f1/fs)  # Starting point of the circle, radius < 1
                    w = np.exp(-1j*np.pi/M)  # "Step size" of circle
                    zchirp = czt(sample.Data[j+1],M + 1, w, a)  # Include Nyquist for comparison to rfft
                    y = zchirp
                    comp_offset = 1e-1
                else:
                    y = sample.Data[j+1]
                    comp_offset = 1e-6

                if magnitude == 'Real':
                    S = custom_stft(y,window=n_fft,delta=hop_length).real
                elif magnitude == 'Imaginary':
                    S = custom_stft(y,window=n_fft,delta=hop_length).imag
                elif magnitude == 'Module':
                    S = np.abs(custom_stft(y,window=n_fft,delta=hop_length))
                elif magnitude == 'Phase':
                    S = np.angle(custom_stft(y,window=n_fft,delta=hop_length))
                    from math import pi
                    S = S + pi
                else:
                    print('Invalid magnitude. Chose between: "Real", "Imaginary", "Module" or "Phase"')
                    exit()
                new_sr = sr

                # Reference states
                if i == 0:
                    ref_S[j] = S
                    base_corr[j] = correlation2D(ref_S[j],ref_S[j],axis=1)

                # Other spectrograms
                CS = correlation2D(ref_S[j],S,axis=1)
                DS = correlation2D(ref_S[j],S,axis=1)-base_corr[j]+comp_offset
                #AC = correlation2D(S,S,axis=1)
                #BC = correlation2D(S,S,axis=1)-base_corr[j]+comp_offset

                for component,E in zip(components,[S,CS,DS]):

                    if feature == 'Spectral centroid':
                        spectral_feature = librosa.feature.spectral_centroid(S = E, sr=new_sr, hop_length=hop_length, n_fft = n_fft)[0]
                    elif feature == 'Spectral rolloff':
                        spectral_feature = librosa.feature.spectral_rolloff(S = E, sr=new_sr, hop_length=hop_length, n_fft = n_fft)[0]
                    elif feature == 'Spectral bandwidth':
                        spectral_feature = librosa.feature.spectral_bandwidth(S = E, sr=new_sr, hop_length=hop_length, n_fft = n_fft)[0]
                    elif feature == 'Spectral contrast':
                        spectral_feature = librosa.feature.spectral_contrast(S = E, sr=new_sr, hop_length=hop_length, n_fft = n_fft)[0]
                    elif feature == 'Spectral flatness':
                        spectral_feature = librosa.feature.spectral_flatness(S = E, hop_length=hop_length, n_fft = n_fft)[0]
                    elif feature == 'RMS':
                        spectral_feature = librosa.feature.rms(S=E, frame_length=int(n_fft*2-1), hop_length=hop_length)[0]

                    z = np.linspace(0,sample.z[-1]-sample.z[0],len(spectral_feature))

                    data[sample_key][state][component] = [z,spectral_feature]
                    if feature == 'RMS':
                        ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'
                    elif feature == 'Spectral flatness':
                        ylabels[sample_key][state] = rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'
                    else:
                        ylabels[sample_key][state] = r'$\nu$ [1/m]'+'\n'+rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'

                type = ''

            elif feature == 'Polyfeatures':

                # Compute STFT
                if magnitude == 'Real':
                    S = custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length).real
                elif magnitude == 'Imaginary':
                    S = custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length).imag
                elif magnitude == 'Module':
                    S = np.abs(custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length))
                elif magnitude == 'Phase':
                    S = np.angle(custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length))
                    from math import pi
                    S = S + pi

                else:
                    print('Invalid magnitude. Choose between: "Real", "Imaginary", "Module" or "Phase"')
                    exit()

                new_sr = sr

                # Reference states
                if i == 0:
                    ref_S[j] = S
                    base_corr[j] = correlation2D(ref_S[j],ref_S[j],axis=1)


                for order in orders:
                    # Spectrogram polyfeatures
                    p = librosa.feature.poly_features(S=S, order=order)
                    z = np.linspace(0,sample.z[-1]-sample.z[0],len(sum(p)))
                    data[sample_key][state][f'Order: {order}'] = [z,sum(p)]

                    # Correlation between signals
                    p = librosa.feature.poly_features(S=correlation2D(ref_S[j],S,axis=1), order=order)
                    z = np.linspace(0,sample.z[-1]-sample.z[0],len(sum(p)))
                    c_data[sample_key][state][f'Order: {order}'] = [z,sum(p)]

                    # Difference between correlation
                    p = librosa.feature.poly_features(S=correlation2D(ref_S[j],S,axis=1)-base_corr[j], order=order)
                    z = np.linspace(0,sample.z[-1]-sample.z[0],len(sum(p)))
                    d_data[sample_key][state][f'Order: {order}'] = [z,sum(p)]
                    check_memory()


                type = ''

            elif feature == 'Tempogram':

                # Compute STFT
                if magnitude == 'Real':
                    S = custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length).real
                elif magnitude == 'Imaginary':
                    S = custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length).imag
                elif magnitude == 'Module':
                    S = np.abs(custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length))
                elif magnitude == 'Phase':
                    S = np.angle(custom_stft(sample.Data[j+1],window=n_fft,delta=hop_length))
                    from math import pi
                    S = S + pi

                new_sr = sr

                oenv = librosa.onset.onset_strength(S=S, sr=new_sr, hop_length=hop_length)
                z = np.linspace(0,sample.z[-1]-sample.z[0],len(oenv))
                oenv_data[sample_key][state]['Onset strength envelope'] = [z,oenv]
                oenv_ylabels[sample_key][state] = ylabels[sample_key][state]

                tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=new_sr,
                                                                hop_length=hop_length)
                t_data[sample_key][state]['Tempogram'] = [tempogram, new_sr]
                t_ylabels[sample_key][state] = 'BPM'+'\n'+ylabels[sample_key][state]


                # Compute global onset autocorrelation
                ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
                ac_global = librosa.util.normalize(ac_global)

                z = np.linspace(0,sample.z[-1]-sample.z[0],len(np.mean(tempogram[1:], axis=1)))
                data[sample_key][state]['Mean local autocorrelation'] = [z, np.mean(tempogram[1:], axis=1)]
                z = np.linspace(0,sample.z[-1]-sample.z[0],len(ac_global[1:]))
                data[sample_key][state]['Global autocorrelation'] = [z,ac_global[1:]]

                type = ''

    # Make the plot
    a_plot(data,ylabels,type=type,linewidth=1)

    # Comprarisson between signals
    c_plot(data,ylabels,type=type,n_fft = 1000,hop_length = 200)

    if feature == 'Polyfeatures':
        a_plot(c_data,ylabels,type=type,linewidth=1)
        a_plot(d_data,ylabels,type=type,linewidth=1)
        c_plot(c_data,ylabels,type=type)
        c_plot(d_data,ylabels,type=type)

    if feature == 'Tempogram':
        a_plot(oenv_data,oenv_ylabels,type=type,linewidth=1)
        a_plot(t_data,t_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'tempo',dB = False)

    plt.show()
