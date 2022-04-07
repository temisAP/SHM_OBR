import scipy.signal as signal

def wavelet(y,plot=False):
    """ Performs a wavelet transform of signal y """
    # define wavelet parameters
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj   # Seven powers of two with dj sub-octaves
    #alpha, _, _ = wavelet.ar1(y)  # Lag-1 autocorrelation for red noise

    #wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, dt, dj, s0, J,
    #                                                      mother)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, dt, dj, s0, J,
                                                         mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother)
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # plot the results
    if plot:
        plot_wavelet(y, wave, iwave, power, fft_power, period, scales,
                     coi, fftfreqs, 'NINO3 SST (degC)', 'NINO3 SST wavelet power spectrum')

    return wave, iwave, power, fft_power, period, scales, coi, fftfreqs

def plot_wavelet(y, wave, iwave, power, fft_power, period, scales,
                 coi, fftfreqs, ylabel, title):
    """ Plots the results of wavelet analysis """
    plt.figure(figsize=(9, 7))
    plt.subplot(221)
    plt.plot(time, y, 'k', label='data')
    plt.xlim(time[0], time[-1])
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.subplot(222)
    plt.imshow(np.log2(power), extent=[time[0], time[-1], 0, max(period)],
               aspect='auto', interpolation='nearest')
    plt.ylim(0, max(period))
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Period (years)')
    plt.title('Wavelet Power Spectrum (in base 2 logarithm)')
    plt.subplot(223)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    plt.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
                 extend='both')
    plt.ylim(0, max(period))
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Period (years)')
    plt.title('Wavelet Power Spectrum (in base 2 logarithm)')
    plt.colorbar()
    plt.subplot(224)
    plt.plot(fftfreqs, np.log2(fft_power), '-', color=[0.7, 0.7, 0.7],
             linewidth=1.)
    plt.plot(fftfreqs, np.log2(fft_power), 'k-', linewidth=1.5)
    plt.ylim(0, 0.5 * np.log2(power.max()))
    plt.xlim(0, 5)
    plt.xlabel('Frequency (cycles/year)')
    plt.ylabel('Power (base 2 logarithm)')
    plt.title('Wavelet Power Spectrum (in base 2 logarithm)')
    plt.tight_layout()
