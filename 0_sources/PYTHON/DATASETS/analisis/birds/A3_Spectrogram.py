import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import librosa.display
from .Z9_utils import arr2librosa, a_plot, b_plot, create_data_and_ylabels, correlation2D, ss_2D, custom_stft
import numpy as np
import scipy


def Spectrogram(samples,n_fft=1000, hop_length= 200,
                    magnitude = 'module',stft='custom_stft',
                    type='normal',
                    cmap='jet'):

    """ Function to draw an spectrogram of the samples given

        : param samples (dict of obrfile objects): samples chosen to be plotted

        : param: n_fft (int):       number of points considered in the window where FFT will be performed
        : param: hop_length (int):  distance between windows, measured in number of points

        : param: magnitude (str):   magnitude plotted in the spectrogram
                                        'module' or 'phase'
        : param: stft (str):        function used to compute STFT
                                        'custom_stft', 'scipy' or 'librosa'
        : param: type (str):        type of spectrogram
                                        'normal', 'mel', 'chroma', 'chroma_cqt' or 'chroma_cens'

    """


    print('\nSpectrogram:')
    print(' magnitude:', magnitude)
    print(' n_fft =',n_fft)
    print(' hop_length =',hop_length)
    print(' stft:', stft)
    print(' type:', type)


    sample_keys = samples.keys()
    states = [r'$P(z)$',r'$S(z)$']
    components = [magnitude]

    data, ylabels       = create_data_and_ylabels(sample_keys,states,components)

    a_data, a_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    b_data, b_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    c_data, c_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    d_data, d_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    #e_data, e_ylabels   = create_data_and_ylabels(sample_keys,states,components)
    #f_data, f_ylabels   = create_data_and_ylabels(sample_keys,states,components)

    ref_data = [[0],[0],[0]]
    base_corr = [[0],[0],[0]]
    #base_2Dcorr = [[0],[0],[0]]

    dB = False


    for i, sample, sample_key in zip(range(len(samples.keys())), samples.values(), samples.keys()):

        Temperature = sample.temperature
        Flecha = sample.flecha
        sr = int(1/(sample.z[1]-sample.z[0]))

        for j, state in enumerate(states):

            # Type of spectrogram
            if type == 'normal':

                # Short-time Fourier transform (STFT)
                if stft == 'librosa':
                    wave, new_sr = arr2librosa(sample.Data[j+1],sr)
                    wave = (wave.real**2 + wave.imag**2)**0.5
                    wave = librosa.stft(wave, n_fft = n_fft, hop_length = hop_length)
                elif stft == 'scipy':
                    f,t,wave = scipy.signal.stft(sample.Data[j+1], fs=sr, nperseg=hop_length,
                                    noverlap=None, nfft=n_fft, detrend=False, return_onesided=True,
                                    boundary=None, padded=True, axis=-1)

                    new_sr = 2*sr
                elif stft == 'custom_stft':
                    wave = custom_stft(sample.Data[j+1], window=n_fft,delta=hop_length)
                    new_sr = sr
                else:
                    print('stft not avaliable. Chose between: "librosa","scipy" or "custom_stft"')

                # Magnitude to consider
                if magnitude == 'module' or magnitude == 'abs':
                    wave = np.abs(wave)
                elif magnitude == 'phase' or magnitude == 'angle':
                    wave = np.angle(wave)


            elif type == 'mel':
                wave, new_sr = arr2librosa(sample.Data[j+1],sr)
                wave = (wave.real**2+wave.imag**2)**0.5

                # Magnitude to consider
                if magnitude == 'module' or magnitude == 'abs':
                    wave = np.abs(wave)
                elif magnitude == 'phase' or magnitude == 'angle':
                    wave = np.angle(wave)

                wave = librosa.feature.melspectrogram(y = wave, sr=new_sr, n_mels=128)
                wave = wave.reshape(wave.shape[0],wave.shape[1]*wave.shape[2]) if len(wave.shape) == 3 else wave

                new_sr = sr/2


            elif type == 'chroma':
                wave, new_sr = arr2librosa(sample.Data[j+1],sr)
                wave = (wave.real**2+wave.imag**2)**0.5

                # Magnitude to consider
                if magnitude == 'module' or magnitude == 'abs':
                    wave = np.abs(wave)
                elif magnitude == 'phase' or magnitude == 'angle':
                    wave = np.angle(wave)

                wave = librosa.feature.chroma_stft(y = wave, sr=new_sr, hop_length=hop_length)

            elif type == 'chroma_cqt':
                print('Under construction')
            elif type == 'chroma_cens':
                print('Under construction')
            else:
                print('Invalid type. Chose between: "normal","mel","chroma","chroma_cqt","chroma_cens"')



            # Reference states
            if i == 0:
                ref_data[j] = wave
                base_corr[j] = correlation2D(ref_data[j],ref_data[j],axis=1)
                #base_2Dcorr[j] = scipy.signal.correlate2d(ref_data[j],ref_data[j], mode='full', boundary='fill', fillvalue=0)

            # Autocorrelation between signals
            a_data[sample_key][state][magnitude] = [correlation2D(wave,wave,axis=1), new_sr]

            # Difference between autocorrelation
            b_data[sample_key][state][magnitude] = [correlation2D(wave,wave,axis=1)-base_corr[j], new_sr]

            # Correlation between signals
            c_data[sample_key][state][magnitude] = [correlation2D(ref_data[j],wave,axis=1), new_sr]

            # Difference between correlation
            d_data[sample_key][state][magnitude] = [correlation2D(ref_data[j],wave,axis=1)-base_corr[j], new_sr]

            # 2D Correlation between signals
            #e_data[sample_key][state][magnitude] = [scipy.signal.correlate2d(ref_data[j], wave, mode='full', boundary='fill', fillvalue=0), new_sr]

            # Difference between 2D correlation
            #f_data[sample_key][state][magnitude] = [scipy.signal.correlate2d(ref_data[j], wave, mode='full', boundary='fill', fillvalue=0)-base_2Dcorr[j], new_sr]


            # Amplitude to dB
            if 'normal' in type and not 'scipy' in stft:
                wave = librosa.amplitude_to_db(wave, ref = np.max)
                dB = True

            # Simple spectrogram
            data[sample_key][state][magnitude]    = [wave, new_sr]
            if type == 'normal' or type == 'mel':
                ylabels[sample_key][state] =  rf'$\nu \: [Hz]$'+'\n'+ rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'
            elif 'chroma' in type:
                ylabels[sample_key][state] =  rf'Pitch class'+'\n'+ rf'$T = {Temperature}\: Cº$'+'\n'+ rf'$\delta = {Flecha}\: mm$'

            a_ylabels[sample_key][state] = b_ylabels[sample_key][state] = ylabels[sample_key][state]
            c_ylabels[sample_key][state] = d_ylabels[sample_key][state] = ylabels[sample_key][state]


    if type == 'normal':
        y_axis = 'log'
        y_axis2 = 'linear'
    elif type == 'mel':
        y_axis = y_axis2 = 'linear'
    elif 'chroma' in type:
        y_axis = y_axis2 = 'chroma'


    # Plot
    a_plot(data,ylabels,hop_length = hop_length, x_axis = 'time', y_axis = y_axis, dB = dB)
    plt.get_current_fig_manager().canvas.set_window_title('Plot')

    # Correlation plot
    a_plot(c_data,c_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = y_axis2,dB = False)
    plt.get_current_fig_manager().canvas.set_window_title('Correlation')

    # Difference between correlations plot
    a_plot(d_data,d_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = y_axis2,dB = False)
    plt.get_current_fig_manager().canvas.set_window_title('Correlation comparisson')

    # Correlation plot
    a_plot(a_data,a_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = y_axis2,dB = False)
    plt.get_current_fig_manager().canvas.set_window_title('Auto correlation')

    # Difference between correlations plot
    a_plot(b_data,b_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = y_axis2,dB = False)
    plt.get_current_fig_manager().canvas.set_window_title('Auto correlation comparisson')

    # Correlation plot
    #a_plot(e_data,e_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'linear',dB = False)

    # Difference between correlations plot
    #a_plot(f_data,f_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = 'linear',dB = False)




    # Possibly birrefringence
    b_plot(data,d_ylabels,hop_length = hop_length, x_axis = 'time', y_axis = y_axis2)
    plt.get_current_fig_manager().canvas.set_window_title('Slow-Fast')

    plt.show()
