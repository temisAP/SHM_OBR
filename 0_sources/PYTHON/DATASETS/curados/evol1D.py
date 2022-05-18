import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
from datetime import datetime
from random import sample


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import find_index

def curing_evol1D(self,points,REF=None,files=None,val='ss',plot=True):

    """ Function get SPECTRAL SHIFT/TEMPERATURE/DEFORMATION along a curing
    process in a given points (up to six)

        : param points      (list)  : list of points x coordinates (in meters) which will be considered

        : optional REF      (str)   : file to consider as reference.    If None the first file created will be used as reference
        : optional files    (list)  : list of files to consider.        If None, all files in 0_OBR folder will be considered
        : optional val      (str)   : variable to display: 'ss', spectral shift; 'temp', temperature; and 'def', deformation
        : optional plot     (bool)  : if True, a plot will be created

        : returns evolution (dict of np.ndarrays) : evolution of the variable along time in each point where the position is the key

    """

    """ Check out """
    self.conditions_checkout()
    self.obr_checkout()

    # Find the earliest file if no ref is specified
    if not REF:

        keys            = list(self.obrfiles.keys())
        earliest_date   = datetime.strptime(self.obrfiles[keys[0]].date,"%Y,%m,%d,%H:%M:%S")
        REF             = self.obrfiles[keys[0]].filename

        for obrfile in self.obrfiles.values():
            file_date = datetime.strptime(obrfile.date,"%Y,%m,%d,%H:%M:%S")
            if file_date < earliest_date:
                earliest_date = file_date
                REF = obrfile.filename

        print(REF,'will be used as reference')

    REF_time = datetime.strptime(self.obrfiles[REF].date,"%Y,%m,%d,%H:%M:%S")

    # Get all files if none is specified
    files = files if files else list(self.obrfiles.keys())
    files.remove(REF) if REF in files else False

    # Compute measures if no measueres computed
    if not hasattr(self, 'measures') or self.measures is None:
        print('\nNo measures found, computing from OBR files...')
        self.obr2measures(REFs=[REF])
        self.save()
        print('Done!')

    # Get all distributions
    val_distributions = list()
    for file in files:
        if val == 'ss':
            val_distribution = self.measures[REF][file].ss
            ylabel = r'$-\frac{\Delta \nu}{\bar{\nu}}$'
        elif val == 'temp':
            val_distribution = self.measures[REF][file].T
            ylabel = r'$\Delta T$[K]'
        elif val == 'def':
            val_distribution = self.measures[REF][file].E
            ylabel = r'$\Delta \mu varepsilon$'

        val_distributions.append(val_distribution)

    z = self.measures[REF][file].x * 1e3 # mm to m

    # Get all the values along the time of all the points specified
    all_vals    =   dict.fromkeys(points)
    all_times   =   dict.fromkeys(points)
    for point in points:

        vals    = list()
        times   = list()
        idx = find_index(z,point)

        for val_distro,file in zip(val_distributions,files):
            vals.append(val_distro[idx])
            file_time = datetime.strptime(self.obrfiles[file].date,"%Y,%m,%d,%H:%M:%S")
            elapsed_time = file_time - REF_time
            times.append(elapsed_time.seconds)

        all_vals[point]     = vals
        all_times[point]    = np.array(times)/60

    if plot:

        plt.figure()
        sample_idxs = sample(range(len(files)),8)
        for idx in sample_idxs:

            file_time = datetime.strptime(self.obrfiles[files[idx]].date,"%Y,%m,%d,%H:%M:%S")
            elapsed_time = file_time - REF_time

            plt.plot(z,val_distributions[idx],label=elapsed_time/60)

        for point in points:
            plt.axvline(point,color='black')

        plt.grid()
        plt.legend()
        plt.xlabel('z [m]')
        plt.ylabel(ylabel,fontsize=20).set_rotation(0)
        plt.show()

        plt.figure()

        for point in points:
            plt.plot(all_times[point],all_vals[point],'o-',label=f'z = {point} m')

        plt.grid()
        plt.legend()
        plt.xlabel('Elapsed time [min]')
        plt.ylabel(ylabel,fontsize=20).set_rotation(0)
        plt.show()

    return all_times,all_vals
