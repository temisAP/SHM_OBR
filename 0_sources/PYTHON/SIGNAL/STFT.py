import scipy.signal as signal


def stft(y,plot=False):
    """
    Compute the stft of the signal y.
    """
    # Parameters: 10ms step, 30ms window
    fs = 44100
    step = int(0.01*fs)
    win_length = int(0.03*fs)
    f, t, Zxx = signal.stft(y, fs=fs, window='hamm', nperseg=win_length, noverlap=win_length-step)
    if plot:
        plot_stft(t, f, Zxx)
    return f, t, Zxx

def plot_stft(t, f, Zxx):
    """
    Plot the stft of the signal y.
    """

    v_max = np.amax(np.array(Zxx))
    v_min = np.amin(np.array(Zxx))

    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)), shading='gouraud',cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=v_min, vmax=v_max))
    cbar = plt.colorbar(sm,spacing='proportional')
