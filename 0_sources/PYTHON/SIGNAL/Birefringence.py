import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
import scipy.optimize as opt
from math import cos, sin, acos, atan2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import printProgressBar


""" Birrefingence

    This module contains functions used to compute the birefringence
    along an optical fiber longitude

"""

def stokes_vector(P,S):

    """ Stokes vector out of P and S polarization states

            : param P (complex 1xn array): p-polarization state
            : param S (complex 1xn array): s-polarization state

            : return [I,Q,U,V] (real 4xn np.array): Stokes vector along z

    """

    n_points = len(P)

    I = list()
    Q = list()
    U = list()
    V = list()

    for n in range(n_points):
        I.append(np.abs(P[n])**2+np.abs(S[n])**2)
        Q.append(np.abs(P[n])**2-np.abs(S[n])**2)
        product = P[n]*np.conj(S[n])
        U.append(2*product.real)
        V.append(2*product.imag)

    return np.array([I,Q,U,V])

def Mueller_matrix(S_in, S_out):

    """ Mueller matrix out of S_in and S_out stokes vectors

            : param S_in  (np.array): Stokes vector before go trought medium
            : param S_out (np.array): Stokes vector after  go trought medium

            : return M (): Stokes vector

    """

    def M_lb(alpha,beta):
        """ Mueller matrix for linear birefringent material """
        M = [   [1,   0,                                              0,                                               0                      ],
                [0,     cos(4*alpha)*(sin(beta/2))**2+(cos(beta/2))**2, sin(4*alpha)*(sin(beta/2))**2,                   sin(2*alpha)*sin(beta) ],
                [0,     sin(4*alpha)*(sin(beta/2))**2,                  -cos(4*alpha)*(sin(beta/2))**2+(cos(beta/2))**2, -cos(2*alpha)*sin(beta)],
                [0,     -sin(2*alpha)*sin(beta),                        cos(2*alpha)*sin(beta),                          sin(beta)              ]]

        return np.array(M)

    def M_pol(theta):
        """ Mueller matrix for ideal polarizer """
        M = [   [1,             cos(2*theta),               sin(2*theta),               0   ],
                [cos(2*theta),  (cos(2*theta))**2 ,         sin(2*theta)*cos(2*theta),  0   ],
                [sin(2*theta),  sin(2*theta)*cos(2*theta),  (sin(2*theta))**2,          0   ],
                [0,             0,                          0,                          0   ]]

        return 0.5 * np.array(M)

    def M_loss(eta):
        """ Mueller matrix for a lossy material """
        M = [  [eta,    0,  0,  0],
                [0,     0,  0,  0],
                [0,     0,  0,  0],
                [0,     0,  0,  0],]

        return np.array(M)


    def eq(x,*args):

        alpha = x[0]
        beta = x[1]
        theta = x[2]
        eta = x[3]

        Mlb   = M_lb(alpha, beta)
        Mpol  = M_pol(theta)
        Mloss = M_loss(eta)
        M = 1/3 * (Mlb + Mpol + Mloss)

        S_in = np.array(args[0])
        S_out = np.array(args[1])
        return S_out - M.dot(S_in)


    # Normalization
    S_in  = 1/S_in[0] * S_in
    S_out = 1/S_out[0] * S_out

    x0 = [0,0,0,1]
    x = opt.fsolve(eq, x0, args=(S_in, S_out))

    alpha = x[0]
    beta = x[1]
    theta = x[2]
    eta = x[3]

    Mlb   = M_lb(alpha, beta)
    Mpol  = M_pol(theta)
    Mloss = M_loss(eta)
    M = 1/3 * (Mlb+Mpol+Mloss)

    return M

def local_birefringence(S1,S2,S3):
    """ Function to compute the mueller matrix over a segment.

        :param  Sn (np.array)   : stokes vectors

        :retruns theta  : delay_angle
    """

    M12 = Mueller_matrix(S1, S2)
    M23 = Mueller_matrix(S2, S3)

    M_delta = 1/4 * (M23 @ np.linalg.inv(M12))

    theta = acos((np.trace(M_delta)-1)/2)

    return theta


def global_birefringence(Data,delta=200,window=1000,display = False,progressbar=False):
    """ Computes relative spectral shift (birefringence) between two signals in a given window
        :param Data     ( list of np.arrays)   : signal
        :param delta    (int)       : index step (optional)
        :param window   (int)       : index window (optional)

        :returns birefringence (np.array)   : array with relative spectral shifts
    """

    P = Data[0]
    S = Data[1]

    delta = 200
    window = 1000

    St = stokes_vector(P,S)
    from scipy.signal import savgol_filter
    St = savgol_filter(St,window,2)

    brf = list()
    n_points = St.shape[1]
    for n in range(n_points-2):
        brf.append(local_birefringence(St[:,n],St[:,n+1],St[:,n+2]))

    brf =  np.array(brf)

    from scipy.signal import savgol_filter
    brf = savgol_filter(brf,delta,1)

    return brf
