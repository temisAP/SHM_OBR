import os
import pandas as pd
import numpy as np
import glob
import re
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from UTILS.utils import find_index
from UTILS.read_obr import multi_read_obr
from UTILS.sensor import sensor
from SIGNAL.filters.optimized import savgol_opt_filter, lfilter_opt_filter, kalman_opt_filter

def obr_filters(self,REF,file=None,delta=2000,window=1000):
    """
    Plots deformation from obr files

        param: REF (str): file which take as reference

        optional: type = None (str):  if 'flecha' plots a colorbar and sorts lines by its deflection
                                      if 'temperature' plots a colorbar and sorts lines by its temperature

        optional: eps = False (bool): if True then adds the deformation along the segment in the plot


    """

    """ Conditions checkout """

    # Conditions file checkout
    conditions_file     = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['conditions filename'])
    if not os.path.isfile(conditions_file):
        print('\nNo conditions file found')
        self.genCONDITIONStemplate()
        return

    conditions_df = pd.read_csv(conditions_file)
    L = float(conditions_df['L\n[mm]'][0])  * 1e-3               # mm to m
    t = float(conditions_df['t\n[mm]'][0])  * 1e-3               # mm to m
    alpha = float(conditions_df['alpha\n[µm/(m·K)]'])            # already microdeformations



    """ Get temperature and deformations from each file and plots it """

    fig, ax = plt.subplots()

    ax.set_ylabel(r'$\Delta \mu\varepsilon$').set_rotation(0)

    if hasattr(self.obrfiles[file], 'Data'):

        refData = self.obrfiles[REF].Data
        Data    = self.obrfiles[file].Data
        z = self.obrfiles[file].z
        f = np.linspace(self.obrfiles[REF].f[0],self.obrfiles[REF].f[-1],3)

        # Add real deformation to graph
        # Get status of the beam
        delta_flecha = (self.obrfiles[file].flecha-self.obrfiles[REF].flecha) * 1e-3             # mm to m
        delta_T      = (self.obrfiles[file].temperature-self.obrfiles[REF].temperature)          # K
        # Relative position on the beam
        x = np.linspace(0,L,len(z))     # in m

        eps_mec = 3*delta_flecha*t/(2*L**3) * (x-L) * 1e6               # Mechanical microdeformations
        eps_the = alpha * delta_T * np.ones_like(x)                     # Thermal  microdeformations
        delta_EPS = eps_mec + eps_the                                   # Total microdeformations

        ax.plot(z,delta_EPS,label='Expected')

        # Create filtered predictions

        TT,EE = sensor(Data,refData,f,delta=delta,window=window,display = False)
        z = np.linspace(z[0],z[-1],len(TT))


        filtereds = {'No-filtered':EE,
                    'Savitzky-Golay':savgol_opt_filter(EE,delta_EPS),
                    'IIR/FIR':lfilter_opt_filter(EE,delta_EPS),         # infinite impulse response (IIR) or finite impulse response (IIR)
                    'Kalman':kalman_opt_filter(EE,delta_EPS)}


        for key,val in filtereds.items():
            ax.plot(z,val,label=key)

        ax.legend()


    ax.set_xlabel('z [m]')
    ax.grid()
    plt.show()
