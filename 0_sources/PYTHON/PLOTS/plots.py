import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../0_sources/PYTHON'))
from UTILS.utils import get_status

def temperature_plot(x,y,lbls,REF):
    """ This functions create a line plot for several datasets,
        if the number of datasets is over 20 it plots a colorbar

        param:    x(np.array):             x axis coordinates (the same for all datasets)
        param:    y(list of np.arrays):    y axis values
        param:    lbls(list of strings):    labels of y axis values
     """

    # Process lbls if possible
    Ts = list()
    new_lbls = list()
    T_ref, Fl_ref  = get_status(REF)
    for lbl in lbls:
        Temperature, Flecha  = get_status(lbl)
        Ts.append(Temperature)
        new_lbls.append(rf'\Delta T$ = {Temperature-T_ref} K | \Delta \mu \delta = {Flecha-Fl_ref}')

    lbls = new_lbls

    # Plot values
    fig, ax = plt.subplots(1,1)
    for yy,lbl in zip(y,lbls):
        ax.plot(x,y, label = lbl) #,color=plt.cm.jet(np.linspace(0,1,len(y))[idx]))

    # Add legend or colorbar

    if len(lbls) <= 20:
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(Ts), vmax=max(Ts)))
        cbar = plt.colorbar(sm)
        if 'grados' in lbl.split('_')[1]:
            cbar.set_label(r'$\Delta T$ [K]',rotation=0,labelpad=10)
        elif 'mm' in lbl.split('_')[1]:
            cbar.set_label(r'$\delta$ [K]',rotation=0,labelpad=10)


    # Add axis labels
    ax.set_xlabel('z [m]')
    ax.set_ylabel(r'$\frac{-\Delta \nu}{\bar{\nu}}$',fontsize=16).set_rotation(0)
    ax.grid()
    plt.tight_layout()
