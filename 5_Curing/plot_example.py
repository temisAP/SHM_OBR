import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib as mpl
import numpy as np
from pynput.mouse import Listener, Button

x = np.linspace(0,100,100)
y = x*1.2+3

fig, ax = plt.subplots()

plt.plot(x, y, 'o')

if True:
    if True:
        if True:

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


plt.show()

print(new_points)
