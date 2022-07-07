import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt
from math import cos, sin
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.read_obr import multi_read_obr
from SIGNAL.Spectral_Shift import global_spectral_shift
from SIGNAL.Birefringence import global_birefringence

def stokes_vector(P,S):

    """ Stokes vector out of P and S polarization states

            : param P (complex number array): p-polarization state
            : param S (complex number array): s-polarization state

            : return [I,Q,U,V] (nxm np.array): Stokes vector along z

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
        M = [   [eta,   0,  0,  0],
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
        M = Mlb + Mpol + Mloss

        S_in = np.array(args[0])
        S_out = np.array(args[1])
        return S_out - M.dot(S_in)


    x0 = [0,0,0,1]
    x = opt.fsolve(eq, x0, args=(S_in, S_out))

    alpha = x[0]
    beta = x[1]
    theta = x[2]
    eta = x[3]

    Mlb   = M_lb(alpha, beta)
    Mpol  = M_pol(theta)
    Mloss = M_loss(eta)
    M = Mlb+Mpol+Mloss

    return M

def mueller_evolution(S):

    S_prima = np.zeros_like(S)
    S_prima[:,0] = S[:,0]

    n_points = S.shape[1]
    for n in range(n_points-1):
        M = Mueller_matrix(S[:,n],S[:,n+1])
        S_prima[:,n+1] = M.dot(S[:,n])

    return S_prima

def Stokes_Mueller(self, REF, files, limit1=0, limit2 = 20, delta = 200, window = 500, plot=True):

    # Take ref
    f,z,Data = multi_read_obr([REF],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1,limit2=limit2)
    refBRF   = global_birefringence(Data[REF],delta=delta,window=window)
    refData  = Data[REF]

    # Calculations
    stokes  = {'I':dict.fromkeys(files), 'Q':dict.fromkeys(files), 'U':dict.fromkeys(files), 'V':dict.fromkeys(files)}
    mueller = {'I':dict.fromkeys(files), 'Q':dict.fromkeys(files), 'U':dict.fromkeys(files), 'V':dict.fromkeys(files)}

    spectralshifts = dict.fromkeys(files)
    birefringences = dict.fromkeys(files)

    for i,file in enumerate(files):
        f,z,Data = multi_read_obr([file],os.path.join(self.path,self.folders['0_OBR']),limit1=limit1,limit2=limit2)

        S = stokes_vector(Data[file][0],Data[file][1])
        from scipy.signal import savgol_filter
        #S = savgol_filter(S,1000,1)
        #S_prima = mueller_evolution(S)

        for idx, key in enumerate(stokes.keys()):
            stokes[key][file]  = S[idx]
            #mueller[key][file] = S_prima[idx]

        spectralshifts[file] = global_spectral_shift(refData[0],Data[file][0],f,delta=delta,window=window)
        birefringences[file] = global_birefringence(Data[file],delta=delta,window=window) - refBRF


    """ Stokes components plot """

    if False:

        """ Stokes vector components """

        fig, ax = plt.subplots(2,2, figsize = (10, 16))

        for i, component in enumerate(stokes.keys()):
                for file in files:
                    ax[i//2,i%2].plot(z,stokes[component][file],label=file)
                ax[i//2,i%2].grid()
                ax[i//2,i%2].set_ylabel(component).set_rotation(0)
                ax[i//2,i%2].set_xlabel('z [m]')


        # Make all y and x axis equal to conserve reference
        for i in range(2):
            for j in range(2):
                ax[0,0].get_shared_x_axes().join(ax[0,0], ax[i,j])
                ax[0,0].get_shared_y_axes().join(ax[0,0], ax[i,j])

        y_limits = [a.get_ylim() for a in  fig.axes[:-1]] ; y_limits = [item for tuple in y_limits for item in tuple]
        plt.setp(ax, ylim=(min(y_limits) , max(y_limits)))

        # Put a general legend at the bottom of the figure
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.figlegend(by_label.values(), by_label.keys(),loc='lower center',ncol=len(files),fancybox=False, shadow=False)

        plt.show()

    """ General lossy linear retarder mueller matrix """

    if False:

        """ Stokes vector components """

        for file in files:

            fig, ax = plt.subplots(2,2, figsize = (10, 16))

            for i, component in enumerate(stokes.keys()):

                    ax[i//2,i%2].plot(z,stokes[component][file], label=file+' true value')
                    ax[i//2,i%2].plot(z,mueller[component][file],label=file+' after mueller')
                    ax[i//2,i%2].grid()
                    ax[i//2,i%2].set_ylabel(component).set_rotation(0)
                    ax[i//2,i%2].set_xlabel('z [m]')


            # Make all y and x axis equal to conserve reference
            for i in range(2):
                for j in range(2):
                    ax[0,0].get_shared_x_axes().join(ax[0,0], ax[i,j])
                    ax[0,0].get_shared_y_axes().join(ax[0,0], ax[i,j])

            y_limits = [a.get_ylim() for a in  fig.axes[:-1]] ; y_limits = [item for tuple in y_limits for item in tuple]
            plt.setp(ax, ylim=(min(y_limits) , max(y_limits)))

            # Put a general legend at the bottom of the figure
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.figlegend(by_label.values(), by_label.keys(),loc='lower center',ncol=len(files),fancybox=False, shadow=False)

            plt.show()


    """ Birefringence and spectralshift """

    if plot:

        """ """

        fig, ax = plt.subplots(1,1, figsize = (10, 16))
        ax2 = ax.twinx()

        for file in files:
            z_ss  = np.linspace(z[0],z[-1],len(spectralshifts[file]))
            z_brf = np.linspace(z[0],z[-1],len(birefringences[file]))
            ax.plot(z_ss,spectralshifts[file],'-o',label=file)
            ax2.plot(z_brf,birefringences[file],label=file)
            ax.grid() ; ax2.grid()
            ax.set_ylabel('Spectral Shift')
            ax2.set_ylabel('Birrefingence')
            ax.set_xlabel('z [m]')


        def align_yaxis(ax1, v1, ax2, v2):
            """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
            _, y1 = ax1.transData.transform((0, v1))
            _, y2 = ax2.transData.transform((0, v2))
            inv = ax2.transData.inverted()
            _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
            miny, maxy = ax2.get_ylim()
            ax2.set_ylim(miny+dy, maxy+dy)

        align_yaxis(ax, 0, ax2, 0)
        ax.legend()
        plt.show()
