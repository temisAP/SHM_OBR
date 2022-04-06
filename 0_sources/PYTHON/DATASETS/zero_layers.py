import numpy as np
import statsmodels.api as sm


def layer0(data,ref_data,f,lamb=25, mode='same'):
    """

    Zero 'layer' for a IA model, it takes signal data and its reference
    and computes autocorrelation comparisson and cross correlation

        param: data (complex 2xn np.array)      : array which contains the signal data
        param: ref_data (comples 2xn np.array)  : array which contains the reference signal data
        param: f (1D np.array)                  : frequency sampling array

            *Both data and ref_data are compound with the same structure:
                    data = [[P],[S]]

        optional: lamb=25       : lambda parameter for the STL filter performed over
                                  autocorrelation comparisson

        optional: mode='same'   : mode for np.correlate function

        returns: X (1D np.array): array which contains cross and auto autocorrelation
                                  information: [[spectralshift],crosscorr,autocorr]

    """

    P1 = data[0]
    S1 = data[1]
    P2 = ref_data[0]
    S2 = ref_data[1]

    """ Autocorrelation """

    # Autocorrelation
    y = P1
    autocorr1 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    y = P2
    autocorr2 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    autocorr = np.absolute(autocorr1-autocorr2)

    # STL filter
    seasonal,trend = np.array(sm.tsa.filters.hpfilter(autocorr, lamb=25))
    autocorr = trend

    autocorr = autocorr[int(len(autocorr)/2-200):int(len(autocorr)/2+200)]

    """ Cross correlation """

    y1 = P1
    y2 = P2

    # Frequency sampling
    DF = f[-1]-f[0]     # Frequency increment
    n = len(y1)         # Sample lenght
    sr = 1/(DF/n)       # Scan ratio

    # FFT
    Y1 = np.absolute(np.fft.fft(y1))
    Y2 = np.absolute(np.fft.fft(y2))
    Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
    Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

    # Cross corelation
    crosscorr = np.correlate(Y1, Y2, mode=mode)

    # Spectral shift
    spectralshift_lags = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)
    spectralshift = spectralshift_lags[np.argmax(crosscorr)]
    spectralshift = -1*spectralshift/np.mean(f)*1e6             # micro spectralshift

    """ Return """

    t = [[spectralshift],crosscorr,autocorr]
    X = np.array([item for sublist in t for item in sublist])

    return X

def layer00(data,ref_data,f,lamb=None, mode=None):

    """

    An alternative for a Zero 'layer' for a IA model, it takes signal data and
    its reference and simply returns all signals in a 1d real array
    after a FFT transformation and signal normalization

        param: data (complex 2xn np.array)      : array which contains the signal data
        param: ref_data (comples 2xn np.array)  : array which contains the reference signal data
        param: f (1D np.array)                  : frequency sampling array

        *Both data and ref_data are compound with the same structure:
                data = [[P],[S]]

        * optional parameters mantained to conserve an easier compatibility

        returns: X (1D np.array): array which contains input information:

                                    [P1.real, P1.imag, S1.real, S1.imag,
                                        P2.real, P2.imag, S2.real, S2.imag]

    """

    P1 = data[0]
    S1 = data[1]
    P2 = ref_data[0]
    S2 = ref_data[1]

    # P transformation and  normalization

    P1 = np.absolute(np.fft.fft(P1))
    P2 = np.absolute(np.fft.fft(P2))
    P1 = (P1 - np.mean(P1)) / (np.std(P1) * len(P1))
    P2 = (P2 - np.mean(P2)) / (np.std(P2))

    # P transformation and normalization

    S1 = np.absolute(np.fft.fft(S1))
    S2 = np.absolute(np.fft.fft(S2))
    S1 = (S1 - np.mean(S1)) / (np.std(S1) * len(S1))
    S2 = (S2 - np.mean(S2)) / (np.std(S2))

    # Return

    t = [P1.real, P1.imag,
        S1.real, S1.imag,
        P2.real, P2.imag,
        S2.real, S2.imag]

    X = np.array([item for sublist in t for item in sublist])

    return X
