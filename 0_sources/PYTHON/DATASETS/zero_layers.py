
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

    """ Frequency sampling """
    DF = f[-1]-f[0]     # Frequency increment
    n = len(P1)         # Sample lenght
    sr = 1/(DF/n)       # Scan ratio
    spectralshift_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)

    secuence1 = [P1,P1,P2,S1,S1,S2]
    secuence2 = [P1,P2,P2,S1,S2,S2]

    Df = 1/sr

    t = list()

    """ Cross correlations """
    for y1,y2 in zip(secuence1,secuence2):

        # FFT
        Y1 = np.absolute(np.fft.fft(y1))
        Y2 = np.absolute(np.fft.fft(y2))
        Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
        Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

        # Cross corelation
        t.extend(np.correlate(Y1, Y2, mode='same').tolist())


    """ Return """
    t = [t,[Df]] # Append frequency step to have information about magnitude
    X = np.array([item for sublist in t for item in sublist])

    return X
