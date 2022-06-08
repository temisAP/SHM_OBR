import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib as mpl
import numpy as np

x = np.linspace(0,100,100)
y = x*1.2+3

fig, ax = plt.subplots()

plt.plot(x, y, 'o')

global points
points = list()

def on_click(event):
    if event.button is MouseButton.LEFT:
        # Create a vertical line on the point clicked
        plt.axvline(x=event.xdata, color='r')
        global points
        points.append(event.xdata)
        plt.draw()
    elif event.button is MouseButton.RIGHT:
        for line in ax.lines:
            if line.get_color() == 'r':
                line.remove()
        points = list()
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

plt.show()

print(points)
