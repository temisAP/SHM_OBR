import os
import pandas as pd
import numpy as np
import glob
import re
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import find_index
from UTILS.read_obr import multi_read_obr
from UTILS.sensor import sensor

def obr_TE(self,REF,files=None,delta=2000,window=1000,type=None,eps=False):
    """
    Plots temperature and deformation from obr files

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

    """ Get filenames to iterate """

    files = list(self.obrfiles.keys()) if files == None else files
    Ts = list()
    files.remove(REF) if REF in files else False
    f = np.linspace(self.obrfiles[REF].f[0],self.obrfiles[REF].f[-1],3)


    """ Get labels """

    for file in files:
        if hasattr(self.obrfiles[file], 'Data'):

            if type == 'temperature':
                Ts.append(self.obrfiles[file].temperature-self.obrfiles[REF].temperature)
            elif type == 'flecha':
                Ts.append(self.obrfiles[file].flecha-self.obrfiles[REF].flecha)

    """ Get temperature and deformations from each file and plots it """

    fig, ax = plt.subplots()

    ax.set_ylabel(r'$\Delta T$'+'\n'+'[K]').set_rotation(0)

    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\Delta \mu\varepsilon$'+'\n'+'(dashed)').set_rotation(0)

    for file in files:
        if hasattr(self.obrfiles[file], 'Data'):

            refData = self.obrfiles[REF].Data
            Data    = self.obrfiles[file].Data
            z = self.obrfiles[file].z

            if type == 'temperature':
                T = int(self.obrfiles[file].temperature-self.obrfiles[REF].temperature)
            elif type == 'flecha':
                T = int(self.obrfiles[file].flecha-self.obrfiles[REF].flecha)

            TT,EE = sensor(Data,refData,f,delta=delta,window=window,display = False)
            z = np.linspace(z[0],z[-1],len(TT))

            # Add real deformation to graph
            if eps:
                # Get status of the beam
                delta_flecha = (self.obrfiles[file].flecha-self.obrfiles[REF].flecha) * 1e-3             # mm to m
                delta_T      = (self.obrfiles[file].temperature-self.obrfiles[REF].temperature)          # K
                # Relative position on the beam
                x = np.linspace(0,L,len(z))     # in m

                eps_mec = 3*delta_flecha*t/(2*L**3) * (x-L) * 1e6               # Mechanical microdeformations
                eps_the = alpha * delta_T * np.ones_like(x)                     # Thermal  microdeformations
                delta_EPS = eps_mec + eps_the                                   # Total microdeformations

                ax2.plot(z,delta_EPS,'--',label=file,color=plt.cm.jet(find_index(Ts,T)/len(Ts)))

            if type == None:
                ax.plot(z,TT,'-' ,label=file+r' $\Delta T$')
                ax.plot(z,EE,'--',label=file+r' $\Delta \mu varepsilon$')
            else:
                ax.plot(z,TT,'-' ,label=file+r' $\Delta T$',color=plt.cm.jet(find_index(Ts,T)/len(Ts)))
                ax.plot(z,EE,'--',label=file+r' $\Delta T$',color=plt.cm.jet(find_index(Ts,T)/len(Ts)))

    # Legend or colorbar
    if type == None:
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(Ts), vmax=max(Ts)))
        cbar = plt.colorbar(sm,ax=ax,spacing='proportional')
        if type == 'temperature':
            cbar.set_label(r'$\Delta T$ [K]',rotation=0,labelpad=15)
        elif type == 'flecha':
            cbar.set_label(r'$\delta$ [mm]',rotation=0,labelpad=15)


    ax.set_xlabel('z [m]')
    ax.grid()
