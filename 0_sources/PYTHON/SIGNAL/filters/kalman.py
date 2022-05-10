# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

# by Andrew D. Straw

import numpy as np
import matplotlib.pyplot as plt

def kalman(z,Q=1e-5,R = 0.1**2,plot=False):
    """ Kalman filter of a given signal

        :param z (np.ndarray): signal which will be filtered

        :optional Q = 1e-5    (float): process variance
        :optional R = 0.1**2  (float): estimate of measurement variance
        :optional plot = False (bool): creates a plot if True

        :return xhat (np.ndarray): filtered signal """


    # intial parameters
    n_iter = len(z)
    sz = (n_iter,)

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor


    # intial guesses
    xhat[0] = 0.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]


    if plot:

        plt.figure()
        plt.plot(z,'k+',label='noisy measurements')
        plt.plot(xhat,'b-',label='a posteri estimate')
        plt.legend()
        plt.show()

    return xhat
