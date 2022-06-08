import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib as mpl
from scipy.interpolate import interp1d
import sys
from datetime import datetime
from random import sample


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import find_index

def curing_evol1D(self,points=None,REF=None,files=None,val='ss',plot=True):

    """ Function get SPECTRAL SHIFT/TEMPERATURE/DEFORMATION along a curing
    process in a given points (up to six)

        : optional points   (list)  : list of points z coordinates (in meters) which will be considered, if none a GUI with selectable segments will be displayed
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
    if not hasattr(self, 'measures') or (self.measures is None) or any([v == None for v in self.measures[REF].values()]):
        print('\nNo measures found, computing from OBR files...')
        self.obr2measures(REFs=[REF])
        self.save()
        print('Done!')

    # Get all distributions
    val_distributions = list()
    time_distribution = list()
    for file in files:
        if val == 'ss':
            val_distribution = self.measures[REF][file].ss
            ylabel = r'$-\frac{\Delta \nu}{\bar{\nu}}$'
        elif val == 'temp':
            val_distribution = self.measures[REF][file].T
            ylabel = r'$\Delta T$[K]'
        elif val == 'def':
            val_distribution = self.measures[REF][file].E
            ylabel = r'$\Delta \mu \varepsilon$'

        val_distributions.append(val_distribution)

        file_time = datetime.strptime(self.obrfiles[file].date,"%Y,%m,%d,%H:%M:%S")
        elapsed_time = file_time - REF_time ; elapsed_time = elapsed_time.total_seconds() / 60
        time_distribution.append(elapsed_time)

    z = self.measures[REF][file].x * 1e3 # mm to m

    # z plot
    if plot:

        fig, ax = plt.subplots()
        max_elapsed_time = REF_time - REF_time; max_elapsed_time = max_elapsed_time.total_seconds() / 60
        for idx,file in enumerate(files):
            if idx%3 == 0:
                file_time = datetime.strptime(self.obrfiles[files[idx]].date,"%Y,%m,%d,%H:%M:%S")
                elapsed_time = file_time - REF_time ; elapsed_time = elapsed_time.total_seconds() / 60  # seconds to minutes
                max_elapsed_time = elapsed_time if elapsed_time > max_elapsed_time else max_elapsed_time
                plt.plot(z,val_distributions[idx],'o',color=plt.cm.jet(find_index(time_distribution,elapsed_time)/len(time_distribution)))

        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max_elapsed_time))
        cbar = plt.colorbar(sm,spacing='proportional')
        cbar.set_label('Elapsed\ntime\n[min]',rotation=0,labelpad=15)
        plt.xlabel('z [m]')
        plt.ylabel(ylabel,fontsize=20,labelpad=20).set_rotation(0) if val == 'ss' else plt.ylabel(ylabel,labelpad=5).set_rotation(0)

        plt.grid()

        if points:
            for point in points:
                plt.axvline(point,linestyle='--',color='black',label='Point chosen')

        else:

            global new_points
            new_points = list()

            def on_click(event):

                if event.button is MouseButton.LEFT:
                    # Create a vertical line on the point clicked
                    plt.axvline(x=event.xdata, linestyle='--',color='black',label='Point chosen')
                    # Update legend box
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
                    # Update list
                    global new_points
                    new_points.append(event.xdata)
                    # Draw
                    plt.draw()

                elif event.button is MouseButton.RIGHT:
                    # Remove all lines
                    for line in ax.lines:
                        if line.get_color() == 'black':
                            line.remove()
                    # Update legend box
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
                    # Delete list
                    new_points.clear()
                    # Draw
                    plt.draw()

            def on_move(event):
                # Update title with the point position
                ax.set_title(f'Point at z = {event.xdata} m')

                # Delete last green line and draw a new one if is into the figure
                for line in ax.lines:
                    if line.get_color() == 'g':
                        line.remove()
                try:
                    plt.axvline(x=event.xdata, color='g')
                    plt.draw()
                except:
                    pass


            def when_leaving_axes(event):
                # Delete the dinamic line line when the mouse leaves the axes
                for line in ax.lines:
                    if line.get_color() == 'g':
                        line.remove()
                plt.draw()

            # Connect the click and move events to the on_click and on_move functions
            fig.canvas.mpl_connect('button_press_event', on_click)
            fig.canvas.mpl_connect('motion_notify_event', on_move)
            fig.canvas.mpl_connect('axes_leave_event', when_leaving_axes)

            x_limits = [a.get_xlim() for a in  fig.axes[:-1]] ; x_limits = [item for tuple in x_limits for item in tuple]
            plt.setp(ax, xlim=(min(x_limits) , max(x_limits)))

            points = new_points

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.show()

    # Compute vals and times
    if points:
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

    # t plot
    if plot:

        plt.figure()

        for point in points:
            # Sort by time
            new_idx = np.argsort(all_times[point])
            all_times[point] = [all_times[point][int(i)] for i in new_idx]
            all_vals[point]  = [all_vals[point][int(i)] for i in new_idx]

            plt.plot(all_times[point],all_vals[point],'o',label=f'z = {point:.3f} m')

        plt.grid()
        plt.legend()
        plt.xlabel('Elapsed time [min]')
        plt.ylabel(ylabel,fontsize=20,labelpad=30).set_rotation(0) if val == 'ss' else plt.ylabel(ylabel,labelpad=5).set_rotation(0)
        plt.show()

    return all_times,all_vals
