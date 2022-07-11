import numpy as np


def NonlinearCorrections(y):

    """ Function to fix the nonlinear light source speed

            param: y  (np.array) : time-domain signal to be correct

            return: y (np.array) : corrected time-domain signal

    """

    # Rectangle band-pass filter was applied on the data in the spatial domain

    y = y

    # IFFT
    #y = np.fft.ifft(Y)

    # Phase calculation

    # Phase unwrap

    # Equal phase interval dividing

    # Resampling

    # Corrected signal

    return y
