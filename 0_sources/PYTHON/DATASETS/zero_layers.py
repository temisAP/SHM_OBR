import numpy as np
import statsmodels.api as sm


def layer0(data,ref_data,z,f,lamb=25, mode='same'):
    """

    Zero 'layer' for a IA model, it takes signal data and its reference
    and computes autocorrelation comparisson and cross correlation

        param: data (complex 2xn np.array)      : array which contains the signal data
        param: ref_data (comples 2xn np.array)  : array which contains the reference signal data
        param: z (1D np.array)                  : length sampling array
        param: f (1D np.array)                  : frequency sampling array


            *Both data and ref_data are compound with the same structure:
                    data = [[P],[S]]

        optional: lamb=25       : lambda parameter for the STL filter performed over
                                  autocorrelation comparisson

        optional: mode='same'   : mode for np.correlate function

        returns: X (1D np.array): array which contains cross and auto autocorrelation
                                  information: [[spectralshift],crosscorr,autocorr]

    """

    # From input data

    P1 = data[0]
    S1 = data[1]
    P2 = ref_data[0]
    S2 = ref_data[1]

    # The output data (will be converted into numpy array)

    X = list()

    # Time and frequency arrays

    DF = f[-1]-f[0]     # Frequency range
    n = len(P1)         # Sample lenght
    sr = 1/(DF/n)       # Scan ratio

    DZ = z[-1]-z[0]     # lenght range
    n = len(P1)         # Sample lenght
    lr = 1/(DZ/n)       # Lengh ratio

    frequency_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)     #[GHz]
    timeshift_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n+1)     #[mm]

    """ ******** Cross correlation (time) ******** """

    # For data

    Y1 = np.absolute(np.array(P1))
    Y2 = np.absolute(np.array(S1))
    print('error') if Y1 is P1 else False
    Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
    Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))
    corr = np.correlate(Y1, Y2, mode='same')

    timeshift = timeshift_arr[np.argmax(corr)]

    X.extend(timeshift, corr[np.argmax(corr)])

    # For reference data

    Y1 = np.absolute(np.array(P2))
    Y2 = np.absolute(np.array(S2))
    print('error') if Y1 is P1 else False
    Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
    Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))
    corr = np.correlate(Y1, Y2, mode='same')

    timeshift = timeshift_arr[np.argmax(corr)]

    X.extend(timeshift, corr[np.argmax(corr)])


    """ ******** Cross correlation (frequency) ******** """

    Y1 = np.absolute(np.fft.fft(P1))
    Y2 = np.absolute(np.fft.fft(P2))
    Y1 = (Y1 - np.mean(Y1)) / (np.std(Y1) * len(Y1))
    Y2 = (Y2 - np.mean(Y2)) / (np.std(Y2))

    spectralshift = frequency_arr[np.argmax(corr)]


    X.extend(spectralshift, corr[np.argmax(corr)])

    """ ******** Autocorrelation comparisson ******** """

    mode = 'same'
    y = P1
    print('error') if Y1 is P1 else False
    autocorr1 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    y = P2
    autocorr2 = np.correlate(y, y, mode=mode)/np.var(y)/len(y)
    autocorr = np.absolute(autocorr1-autocorr2)

    X.extend(timeshift_arr[np.argmax(autocorr)], autocorr[np.argmax(autocorr)])
    X.extend(timeshift_arr[np.argmin(autocorr)], autocorr[np.argmin(autocorr)])
    X.extend(timeshift_arr[int(len(autocorr)/2)],reference_value)


    return np.array(X)
