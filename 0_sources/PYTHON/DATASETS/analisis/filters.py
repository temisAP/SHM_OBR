import os
import pandas as pd
import numpy as np
import glob
import re
import sys
import time
import matplotlib.pyplot as plt
import random


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from UTILS.utils import find_index, check_memory
from UTILS.read_obr import multi_read_obr
from SIGNAL.filters.optimized import optimize


def obr_fit_filters(self,REF,type='deformation',samples=1,delta=2000,window=1000):
    """
    Fit filters for all obr files

        param: REF (str): file which take as reference to make predictions about status change

        optional: delta = 2000 (int)
        optional: window = 1000 (int)


    """

    filters_to_fit = [
                        'savgol',
                        'lfilter',
                        'kalman',
                        'stl']
    # Initialize atribute in self if not already initialized

    if not hasattr(self,'filters'):
        keys = list()
        for filter in filters_to_fit:
            keys.append(f'{filter}_deformation')
            keys.append(f'{filter}_temperature')
        self.filters = dict.fromkeys(keys)

    # Make predictions

    files = random.sample(self.obrfiles.keys(),samples)
    print('Files to train:',files)

    predictions = list()
    targets = list()

    for file in files:
        check_memory()
        prediction, target = self.obr_filters(REF,file,type=type,delta=delta,filters=None,plot=False)
        predictions.append(prediction)
        targets.append(target)

    # Fit filters
    for filter in filters_to_fit:
        check_memory()
        self.filters[f'{filter}_{type}'] = optimize(predictions,targets,filter,return_obj = True)

    # Show results
    file = files[0] #file = random.sample(self.obrfiles.keys(),1)[0]
    print('File to test:',file)
    self.obr_filters(REF,file,type=type,delta=delta)

def obr_filters(self,REF,file,
                type='deformation',delta=2000,window=1000,plot=True,
                filters={
                    'Savitzky-Golay': 'savgol',
                    'IIR/FIR'       : 'lfilter',
                    'Kalman'        : 'kalman',
                    'STL (trend)'   : 'stl'}):

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

    if plot:
        fig, ax = plt.subplots()
        ax.set_ylabel(r'$\Delta \mu\varepsilon$' if type == 'deformation' else r'$\Delta \: T$' ).set_rotation(0)

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
        delta_T = delta_T * np.ones_like(x)

        target = delta_EPS if type == 'deformation' else delta_T

        ax.plot(z, target, label='Expected') if plot else False

        # Create filtered predictions

        TT,EE = self.sensor(Data,refData,f,delta=delta,window=window,display = False)
        z = np.linspace(z[0],z[-1],len(TT))

        prediction = EE if type == 'deformation' else TT

        if filters:
            filtereds = dict.fromkeys(filters)
            filtereds['No-filtered prediction'] = prediction

            for key,val in filters.items():
                filtereds[key] = self.filters[f'{val}_{type}'].filter_this(prediction)

            for key,val in filtereds.items():
                ax.plot(z,val,label=key) if plot else False

        ax.legend() if plot else False

    if plot:
        ax.set_xlabel('z [m]')
        ax.grid()
        plt.show()
    else:
        return prediction, target
