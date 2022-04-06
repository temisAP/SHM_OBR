import os
import pandas as pd
import matplotlib.pyplot as plt

def dataset_plot(self):

    """ Plots dataset spectralshift along x to check if slices treatment has been correct """

    info = pd.read_csv(
        os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['dataset book filename']))

    to_plot = ['d_flecha', 'delta_EPS', 'delta_T']

    for val in to_plot:
        plt.figure()
        plt.title(val)
        plt.scatter(info['x'],info['spectralshift'],c=info[val],cmap='jet')
        plt.grid()
        plt.colorbar()

    plt.show()
