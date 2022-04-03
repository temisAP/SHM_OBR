import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

class a_scaler(object):
    def __init__(self,max,min):
        self.max = max
        self.min = min
        if self.max == self.min:
            print('Scaler is broken :(')
            exit()

    def transform(self,x):
        return (x-self.min)/(self.max-self.min)

    def inverse_transform(self,z):
        return (self.max-self.min) * z + self.min


def pre_processing(self,X,y,plot_preprocessing=False,plot_histogram=False):

    # Lists to np.arrays

    for key,val in X.items():
        X[key] = np.array(X[key])
        y[key] = np.array(y[key])

    # Set up scalers

    xx = X['train'].flatten().reshape(-1, 1)
    self.scalerX = scalerX = a_scaler(xx.max(),xx.min())

    xx = y['train'][:,0].flatten().reshape(-1, 1)
    self.scaler_T  = scaler_T  = a_scaler(xx.max(),xx.min())
    xx = y['train'][:,1].flatten().reshape(-1, 1)
    self.scaler_E  = scaler_E  = a_scaler(xx.max(),xx.min())

    # Transform each sample

    for key,val in X.items():

        # Transform each sample
        X[key] = np.array([scalerX.transform(sample) for sample in X[key]])

        T      = np.array([scaler_T.transform(sample)  for sample in y[key][:,0]     ])
        E      = np.array([scaler_E.transform(sample)  for sample in y[key][:,1]     ])
        y[key] = np.concatenate((np.transpose(np.array([T])),np.transpose(np.array([E]))),axis=1)


        # If plot, then plot ss and ac before and after normalization
        if plot_preprocessing:
            for idx in range(5):
                fig, axs = plt.subplots(2, 2, constrained_layout=True)

                axs[0,0].plot(X[key][idx,1:2001])
                axs[0,0].grid()
                axs[0,0].set_title('CC Before')

                axs[0,1].plot(cc[idx,:])
                axs[0,1].grid()
                axs[0,1].set_title('CC After')

                axs[1,0].plot(X[key][idx,-400:])
                axs[1,0].grid()
                axs[1,0].set_title('AC Before')

                axs[1,1].plot(ac[idx,:])
                axs[1,1].grid()
                axs[1,1].set_title('AC After')

                fig.suptitle(f'Normalization {idx}')

        # Plot histograms of values of x, T and E
        if plot_histogram:

            fig, axs = plt.subplots(3, 1, constrained_layout=True)

            axs[0].hist(X[key].flatten().reshape(-1, 1),bins=20)
            axs[0].set_xlim(-0.5,1.5)
            axs[0].grid()
            axs[0].set_title('x')

            axs[1].hist(y[key][:,0],bins=20)
            axs[1].set_xlim(-0.5,1.5)
            axs[1].grid()
            axs[1].set_title('T')

            axs[2].hist(y[key][:,1],bins=20)
            axs[2].set_xlim(-0.5,1.5)
            axs[2].grid()
            axs[2].set_title('E')

            fig.suptitle(f'Histograms for {key}')

        plt.show()

    return X,y
