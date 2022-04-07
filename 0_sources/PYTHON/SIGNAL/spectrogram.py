import scipy.signal as signal

def spectrogram(y,plot=False):
    """
    Compute the spectrogram of the signal y.
    """
    f, t, Sxx = signal.spectrogram(y, fs=1.0, window='hamming',
                                   nperseg=512, noverlap=256,
                                   detrend=False, scaling='spectrum')

    if plot:
        plot_spectrogram(t, f, Sxx)
    elif plot == 'log':
        plot_spectrogram(t, f, 10*np.log10(Sxx))

    return f, t, Sxx

def plot_spectrogram(t, f, Sxx):
    """
    Plot the spectrogram of the signal y.
    """

    v_max = np.amax(np.array(Sxx))
    v_min = np.amin(np.array(Sxx))

    plt.pcolormesh(t, f, Sxx, vmin=v_min,vmax=v_max, shading='gouraud',cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=v_min, vmax=v_max))
    cbar = plt.colorbar(sm,spacing='proportional')
