import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from random import sample

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl

from matplotlib.backend_bases import MouseButton
from pynput.mouse import Listener, Button

from scipy.interpolate import interp1d

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from UTILS.utils import find_index

def curing_evol1D(self,points=None,REF=None,files=None,
                    val='ss',plot=True,
                    base_marker = 'v',marker_size=0.75, markermap = ['o','v','D','X','^','s','P'],
                    t='time',t_limits = None,colorIS = 'Elapsed\ntime\n[min]'):

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

    # Determine if there are more than one evolution -> Filename contains EvoX_
    if all(['Evo' in n for n in self.obrfiles.keys()]):
        print('More than one evolution found')
        t_pos = 1
    else:
        t_pos = 0


    # Find the earliest file if no ref is specified
    if not REF:

        keys            = list(self.obrfiles.keys())
        earliest_date   = datetime.strptime(self.obrfiles[keys[0]].date,"%Y,%m,%d,%H:%M:%S") if t=='time' else float(self.obrfiles[keys[0]].name.split('_')[t_pos])
        REF             = self.obrfiles[keys[0]].filename

        for obrfile in self.obrfiles.values():
            file_date = datetime.strptime(obrfile.date,"%Y,%m,%d,%H:%M:%S") if t=='time' else float(obrfile.name.split('_')[t_pos])
            if file_date < earliest_date:
                earliest_date = file_date
                REF = obrfile.filename

        print(REF,'will be used as reference')

    REF_time = datetime.strptime(self.obrfiles[REF].date,"%Y,%m,%d,%H:%M:%S") if t=='time' else float(self.obrfiles[REF].name.split('_')[t_pos])

    # Get all files if none is specified
    files = files if files else list(self.obrfiles.keys())
    if t_pos == 1:
        files = sort_evo(files)
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
            #val_distribution = self.measures[REF][file].E
            val_distribution = self.measures[REF][file].ss * 189394.17022471415 * -6.6680 * -8.80840637e2/119.88246527477283
            ylabel = r'$\Delta \mu \varepsilon$'

        val_distributions.append(val_distribution)

        if t=='time':
            file_time = datetime.strptime(self.obrfiles[file].date,"%Y,%m,%d,%H:%M:%S")
            elapsed_time = file_time - REF_time ; elapsed_time = elapsed_time.total_seconds() / 60
        else:
            file_time = float(self.obrfiles[file].name.split('_')[t_pos])
            elapsed_time = file_time - REF_time
        time_distribution.append(elapsed_time)

    z = self.measures[REF][file].x * 1e3 # mm to m

    # z plot
    if plot:

        fig, ax = plt.subplots()
        if not t_limits and t=='time':
            max_elapsed_time = REF_time - REF_time; max_elapsed_time = max_elapsed_time.total_seconds() / 60 # sconds to minutes
            min_elapsed_time = REF_time - REF_time; min_elapsed_time = min_elapsed_time.total_seconds() / 60 # sconds to minutes
        elif not t_limits and t!='time':
            max_elapsed_time = REF_time - REF_time
            min_elapsed_time = REF_time - REF_time
        elif t_limits:
            max_elapsed_time = t_limits[1]
            min_elapsed_time = t_limits[0]

        if len(files)>= 100:
            leap = 3
        else:
            leap = 1

        for idx,file in enumerate(files):
            if idx%leap == 0:
                file_time = datetime.strptime(self.obrfiles[files[idx]].date,"%Y,%m,%d,%H:%M:%S") if t=='time' else float(self.obrfiles[files[idx]].name.split('_')[t_pos])
                elapsed_time = file_time - REF_time ; elapsed_time = elapsed_time.total_seconds() / 60 if t=='time' else elapsed_time
                if not t_limits:
                    max_elapsed_time = elapsed_time if elapsed_time > max_elapsed_time else max_elapsed_time
                    min_elapsed_time = elapsed_time if elapsed_time < min_elapsed_time else min_elapsed_time

        for idx,file in enumerate(files):
            if idx%leap == 0:
                file_time = datetime.strptime(self.obrfiles[files[idx]].date,"%Y,%m,%d,%H:%M:%S") if t=='time' else float(self.obrfiles[files[idx]].name.split('_')[t_pos])
                elapsed_time = file_time - REF_time ; elapsed_time = elapsed_time.total_seconds() / 60 if t=='time' else elapsed_time
                plt.plot(z,val_distributions[idx],base_marker,markersize=marker_size,color=plt.cm.jet((elapsed_time-min_elapsed_time)/(max_elapsed_time-min_elapsed_time)))

        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min_elapsed_time, vmax=max_elapsed_time))
        cbar = plt.colorbar(sm,spacing='proportional')
        cbar.set_label(colorIS,rotation=0,labelpad=15)
        plt.xlabel('z [m]')
        plt.ylabel(ylabel,fontsize=15,labelpad=20).set_rotation(0) if val == 'ss' else plt.ylabel(ylabel,labelpad=5).set_rotation(0)

        plt.grid()

        if not points is None:
            for point in points:
                if isinstance(point,float):
                    plt.axvline(point,linestyle='--',color='black',label='Point chosen')
                elif isinstance(point,list):
                    plt.axvline(x=point[0], linestyle='-', linewidth=2,color='tab:orange',label='Interval edges')
                    plt.axvline(x=point[1], linestyle='-', linewidth=2,color='tab:orange',label='Interval edges')
                    plt.axvspan(point[0], point[1], alpha=0.3, color='tab:orange', label='Interval choosen')


        else:
            global new_points
            new_points = list()
            global start_point
            start_point = 0
            global end_point
            end_point = 0
            global key_status
            key_status = ''
            global aspan_list
            aspan_list = []
            global dragging
            dragging = False

            def on_click(x, y, button, pressed):
                if button == Button.left:
                    global key_status
                    key_status = 'pressed' if pressed else 'released'

            listener = Listener(on_click = on_click)
            listener.start()

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
                        if line.get_color() == 'black' or line.get_color() == 'tab:orange':
                            line.remove()
                    # Remove all axvspans
                    global aspan_list
                    for aspan in aspan_list:
                        if aspan.__dict__['_original_facecolor'] == 'tab:orange':
                            aspan.remove()
                            aspan_list.remove(aspan)
                    # Update legend box
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
                    # Delete list
                    new_points.clear()
                    # Draw
                    plt.draw()

            def on_move(event):

                global new_points
                global end_point
                global start_point
                global key_status
                global dragging

                # Update title with the point position
                try:
                    ax.set_title(f'Point at z = {float(event.xdata):.5f} m')
                except:
                    ax.set_title('')

                # Delete last green line and draw a new one if is into the figure
                for line in ax.lines:
                    if line.get_color() == 'g':
                        line.remove()
                # Try drawing new one (to avoid warning when out of box)
                try:
                    plt.axvline(x=event.xdata, color='g')
                except:
                    pass

                # If control key is stroked start listening

                if key_status == 'pressed':

                    # Now we are dragging
                    dragging = True

                    # Find last black line
                    line = ax.lines[-2]
                    if line.get_color() == 'black':
                        # Get its position
                        start_point = line.get_xdata()[0]
                        # Make it green
                        line.set_color("tab:green")
                        # Make it continuous
                        line.set_linestyle('-')
                        # Make it thicker
                        line.set_linewidth(2)

                    # Get current position
                    end_point = float(event.xdata)

                    # Delete last area drawn
                    try:
                        if aspan_list[-1].__dict__['_original_facecolor'] == 'tab:green':
                            aspan_list[-1].remove()
                            aspan_list.remove(aspan_list[-1])
                    except Exception as e:
                        #print(e)
                        pass

                    # Draw newarea
                    aspan_list.append(plt.axvspan(start_point, end_point, alpha=0.3, color='tab:green'))

                    # Update
                    plt.draw()

                elif key_status == 'released' and dragging == True:

                    # Now we have stop dragging
                    dragging = False

                    # Delete last temporal area drawn
                    try:
                        if aspan_list[-1].__dict__['_original_facecolor'] == 'tab:green':
                            aspan_list[-1].remove()
                    except Exception as e:
                        #print(e)
                        pass

                    # Print definitive area
                    plt.axvline(x=start_point, linestyle='-', linewidth=2,color='tab:orange',label='Interval edges')
                    plt.axvline(x=end_point, linestyle='-', linewidth=2,color='tab:orange',label='Interval edges')
                    aspan_list.append(plt.axvspan(start_point, end_point, alpha=0.3, color='tab:orange', label='Interval choosen'))

                    # Find green lines (temporal) and make it orange (fixed)
                    for line in ax.lines:
                        if line.get_color() == 'tab:green':
                            # Make it orange
                            line.set_color("tab:orange")
                            # Make it continuous
                            line.set_linestyle('-')
                            # Make it thicker
                            line.set_linewidth(2)
                            # Make label none
                            line.set_label(None)

                    # Replace start point with a list

                    from copy import deepcopy
                    new_points[-1] = [deepcopy(start_point), deepcopy(end_point)]
                    # Restart key
                    key_status = ''

                    # Update legend box
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())

                elif key_status == 'released' and dragging == False:
                    key_status = ''

                plt.draw()

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
        plt.ylim(-2000,5000)

        plt.tight_layout()
        plt.show()

        print('\nPoints choosen:')
        print(' ',points)

    # Compute vals and times
    if isinstance(points,list):

        # Create dictionary with its keys
        points_labels = list()
        for point in points:
            if isinstance(point,float):
                points_labels.append(point)
            elif isinstance(point,list):
                points_labels.append((point[1]+point[0])/2)

        all_vals    =   dict.fromkeys(points_labels)
        all_times   =   dict.fromkeys(points_labels)

        for point_label,point in zip(points_labels,points):

            vals    = list()
            times   = list()
            idx     = find_index(z,point)

            for val_distro,file in zip(val_distributions,files):
                if isinstance(idx,int) or isinstance(idx,np.integer):
                    vals.append(val_distro[idx])
                elif isinstance(idx,list):
                    # Get all values in segment
                    idx.sort()
                    data = np.array(val_distro[idx[0]:idx[1]])
                    # Delete marginal values
                    q1 = np.percentile(data,25)
                    q3 = np.percentile(data,75)
                    data = [val for val in data if val>=q1 and val<=q3]
                    vals.append(data)

                file_time = datetime.strptime(self.obrfiles[file].date,"%Y,%m,%d,%H:%M:%S") if t=='time' else float(self.obrfiles[file].name.split('_')[t_pos])
                elapsed_time = file_time - REF_time; elapsed_time = elapsed_time.total_seconds() / 60 if t=='time' else elapsed_time
                times.append(elapsed_time)

            all_vals[point_label]     = vals
            all_times[point_label]    = times
    else:
        print('No points found')

    # t plot
    if plot:

        # Manage colormap
        if len(points) <= 10:
            colormap = cm.get_cmap('tab10')
        else:
            colormap = cm.get_cmap('rainbow')
        i = 0


        plt.figure()

        for point_label,point in zip(points_labels,points):
            # Sort by time
            if t == 'time':
                new_idx = np.argsort(all_times[point_label])
                all_times[point_label] = [all_times[point_label][int(i)] for i in new_idx]
                all_vals[point_label]  = [all_vals[point_label][int(i)] for i in new_idx]

            if isinstance(all_vals[point_label][0],float):
                plt.plot(all_times[point_label],all_vals[point_label],
                    markermap[i-1],label=f'z = {point_label:.3f} m',color=colormap(i)); i+=1
            else:
                c =colormap(i);i+=1
                mu = list()
                for data in all_vals[point_label]:
                    # Get mean distribution
                    mu.append(np.mean(data))

                # Append a list
                line = plt.plot(all_times[point_label],mu,markermap[i-1],label=f'Segment no{i}',color=c)

        plt.grid()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.xlabel(colorIS.replace('\n',' '))
        plt.ylabel(ylabel,fontsize=15,labelpad=20).set_rotation(0) if val == 'ss' else plt.ylabel(ylabel,labelpad=5).set_rotation(0)
        #plt.xlim(0,40)
        #plt.ylim(-6e-5,0.99e-5)
        plt.tight_layout()
        plt.show()

    # t-selectable plot
    if plot:


        # Manage colormap
        if len(points) <= 10:
            colormap = cm.get_cmap('tab10')
        else:
            colormap = cm.get_cmap('rainbow')
        i = 0

        # To get and avoid reference segment
        j = 0
        ref = list()
        mu_diff   = dict.fromkeys(points_labels)

        fig, ax = plt.subplots()

        for point_label,point in zip(points_labels,points):

            if j == 0:
                ref_evolution = np.array(all_vals[point_label])
                j += 1

            c =colormap(i);i+=1
            mu_diff[point_label]  = list()
            for data,ref_data in zip(all_vals[point_label],ref_evolution):
                # Get mean distribution
                #mu_diff[point_label].append(np.abs(np.mean(data)-np.mean(ref_data)))
                mu_diff[point_label].append(np.mean(data))

            # Plot
            plt.plot(all_times[point_label],mu_diff[point_label],markermap[i-1],  label=rf'z = {point_label:.3f} m',color=c)

        if True:

            global interest_interval
            interest_interval = list()
            global start_time
            start_time = 0
            global end_time
            end_time = 0
            global key_status2
            key_status2 = ''
            global aspan_list2
            aspan_list2 = []
            global dragging2
            dragging2 = False

            def on_click(x, y, button, pressed):
                if button == Button.left:
                    global key_status2
                    key_status2 = 'pressed' if pressed else 'released'

            listener = Listener(on_click = on_click)
            listener.start()

            def on_click(event):

                if event.button is MouseButton.LEFT:
                    # Create a vertical line on the Time clicked
                    plt.axvline(x=event.xdata, linestyle='--',color='black',label='Time chosen')
                    # Update legend box
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
                    # Update list
                    global interest_interval
                    interest_interval.append(event.xdata)
                    # Draw
                    plt.draw()

                elif event.button is MouseButton.RIGHT:
                    # Remove all lines
                    for line in ax.lines:
                        if line.get_color() == 'black' or line.get_color() == 'tab:orange':
                            line.remove()
                    # Remove all axvspans
                    global aspan_list2
                    for aspan in aspan_list2:
                        if aspan.__dict__['_original_facecolor'] == 'tab:orange':
                            aspan.remove()
                            aspan_list2.remove(aspan)
                    # Update legend box
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
                    # Delete list
                    interest_interval.clear()
                    # Draw
                    plt.draw()

            def on_move(event):

                global interest_interval
                global end_time
                global start_time
                global key_status2
                global dragging2

                # Update title with the Time position
                try:
                    ax.set_title(f'Point at t = {float(event.xdata):.5f} s') if t=='time' else ax.set_title(f'Point at {float(event.xdata):.5f} [{t}]')
                except:
                    ax.set_title('')

                # Delete last green line and draw a new one if is into the figure
                for line in ax.lines:
                    if line.get_color() == 'g':
                        line.remove()
                # Try drawing new one (to avoid warning when out of box)
                try:
                    plt.axvline(x=event.xdata, color='g')
                except:
                    pass

                # If control key is stroked start listening

                if key_status2 == 'pressed':

                    # Now we are dragging2
                    dragging2 = True

                    # Find last black line
                    line = ax.lines[-2]
                    if line.get_color() == 'black':
                        # Get its position
                        start_time = line.get_xdata()[0]
                        # Make it green
                        line.set_color("tab:green")
                        # Make it continuous
                        line.set_linestyle('-')
                        # Make it thicker
                        line.set_linewidth(2)

                    # Get current position
                    end_time = float(event.xdata)

                    # Delete last area drawn
                    try:
                        if aspan_list2[-1].__dict__['_original_facecolor'] == 'tab:green':
                            aspan_list2[-1].remove()
                            aspan_list2.remove(aspan_list2[-1])
                    except Exception as e:
                        #print(e)
                        pass

                    # Draw newarea
                    aspan_list2.append(plt.axvspan(start_time, end_time, alpha=0.3, color='tab:green'))

                    # Update
                    plt.draw()

                elif key_status2 == 'released' and dragging2 == True:

                    # Now we have stop dragging2
                    dragging2 = False

                    # Delete last temporal area drawn
                    try:
                        if aspan_list2[-1].__dict__['_original_facecolor'] == 'tab:green':
                            aspan_list2[-1].remove()
                    except Exception as e:
                        #print(e)
                        pass

                    # Print definitive area
                    plt.axvline(x=start_time, linestyle='-', linewidth=2,color='tab:orange',label='Interval edges')
                    plt.axvline(x=end_time, linestyle='-', linewidth=2,color='tab:orange',label='Interval edges')
                    aspan_list2.append(plt.axvspan(start_time, end_time, alpha=0.3, color='tab:orange', label='Interval choosen'))

                    # Find green lines (temporal) and make it orange (fixed)
                    for line in ax.lines:
                        if line.get_color() == 'tab:green':
                            # Make it orange
                            line.set_color("tab:orange")
                            # Make it continuous
                            line.set_linestyle('-')
                            # Make it thicker
                            line.set_linewidth(2)
                            # Make label none
                            line.set_label(None)

                    # Replace start Time with a list

                    from copy import deepcopy
                    interest_interval[-1] = [deepcopy(start_time), deepcopy(end_time)]
                    # Restart key
                    key_status2 = ''

                    # Update legend box
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())

                elif key_status2 == 'released' and dragging2 == False:
                    key_status2 = ''

                plt.draw()

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
            #plt.setp(ax, xlim=(min(x_limits) , max(x_limits)))

        plt.grid()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.xlabel(colorIS.replace('\n',' '))
        plt.ylabel(ylabel,fontsize=15,labelpad=20).set_rotation(0) if val == 'ss' else plt.ylabel(ylabel,labelpad=5).set_rotation(0)
        plt.show()

        print('\nIntervals of interest:')
        print(' ',interest_interval)

    # Compute differences in the interest intervals
    if isinstance(interest_interval,list):

        all_diffs = dict.fromkeys(points_labels)
        print('\nMean diference between mean differences')

        # Show a all_diffs box and whiskers plot for each interval
        for interval in interest_interval:

            plt.figure()
            i = 0
            if isinstance(interval,list):
                for point_label in points_labels:
                    # Get index
                    idxs = find_index(all_times[point_label],interval)
                    data = mu_diff[point_label][idxs[0]:idxs[1]]
                    # Get color and position
                    c =colormap(i);i+=1
                    # Delete marginal values
                    q1 = np.percentile(data,25)
                    q3 = np.percentile(data,75)
                    data = [val for val in data if val>=q1 and val<=q3]

                    all_diffs[point_label]  = np.mean(data)
                    box = plt.boxplot(data,positions = [i], manage_ticks=True,
                                widths=(1.2),
                                patch_artist=True,
                                showfliers = False,
                                boxprops=dict(facecolor=c, color=c, alpha=0.5),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c))

                plt.xticks([y+1 for y in range(len(points_labels))], [rf'{round(pl,3)}' for pl in points_labels])
                plt.xlabel('z [m]')
                plt.ylabel(fr'$\mu$('+ylabel+')',fontsize=15,labelpad=20).set_rotation(0)
                plt.grid()
                plt.show()

                print(' For interval between:',interval,'min')
                print(' ',[d[1] for d in all_diffs.items()])

            else:
                for point_label in points_labels:
                    # Get index
                    idx = find_index(all_times[point_label],interval)
                    data = mu_diff[point_label][idx]
                    all_diffs[point_label]  = data

                print(' For time:',interval,'min')
                print(' ',[d[1] for d in all_diffs.items()])


    return all_times,all_vals


def sort_evo(files):
    """
    Funtion to sort filenames formated as: "EvoX_NN"
    first by its evolution (X) then by the number after "_" (NN)

        param:  files (list of string) : list of original filenames
        return: files (list of string) : list of sorted filenames

    """
    files.sort(key=lambda x: (int(x.split("_")[0].replace("Evo",'')), float(x.split("_")[1].replace('.obr',''))))
    return files
