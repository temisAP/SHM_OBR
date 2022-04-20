import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

class a_scaler(object):
    def __init__(self,max,min):
        self.max = max
        self.min = min
        if self.max == self.min:
            print('Scaler is broken :(')
            self.max = 1
            self.min = 0

    def transform(self,x):
        return (x-self.min)/(self.max-self.min)

    def inverse_transform(self,z):
        return (self.max-self.min) * z + self.min


def pre_processing(self,X,y,plot_preprocessing=False,plot_histogram=False):

    # Lists to np.arrays

    for key,val in X.items():
        X[key] = np.array(X[key])
        y[key] = np.array(y[key])
        print(f'{key} size = {len(X[key])}')

    # Set up scalers


    xx = y['train'][:,0].flatten().reshape(-1, 1)
    self.scaler_T  = scaler_T  = a_scaler(xx.max(),xx.min())

    xx = y['train'][:,1].flatten().reshape(-1, 1)
    self.scaler_E  = scaler_E  = a_scaler(xx.max(),xx.min())

    # Transform each sample

    for key,val in X.items():

        # Transform each sample
        T  = np.array([scaler_T.transform(sample)  for sample in y[key][:,0]     ])
        E  = np.array([scaler_E.transform(sample)  for sample in y[key][:,1]     ])

        # Plot histograms of values of ss, cc, ac, T and E
        if plot_histogram:

            fig, axs = plt.subplots(4, 4, constrained_layout=True)

            for idx,val in enumerate(np.transpose(X[key])):

                axs[idx//4,idx%4].hist(val,bins=20)
                axs[idx//4,idx%4].set_xlim(-0.5,1.5)
                axs[idx//4,idx%4].grid()
                axs[idx//4,idx%4].set_title(idx)

            fig.suptitle(f'Histograms for {key}: inputs')

            fig, axs = plt.subplots(1, 3, constrained_layout=True)

            axs[0].hist(T,bins=20)
            axs[0].set_xlim(-0.5,1.5)
            axs[0].grid()
            axs[0].set_title('T')

            axs[1].hist(E,bins=20)
            axs[1].set_xlim(-0.5,1.5)
            axs[1].grid()
            axs[1].set_title('E')

            fig.suptitle(f'Histograms for {key}: outputs')

        plt.show()

        # Concatenate information
        X[key] = X[key]
        y[key] = np.concatenate((np.transpose(np.array([T])),np.transpose(np.array([E]))),axis=1)

    return X,y
