import tkinter as tk
from tkinter import filedialog
import numpy as np
import os

def fiber_distribution(self,LX=300,LY=150):

    # Set scale, and borders
    scale = 5 # px/mm

    COLS = LX*scale
    ROWS = LY*scale

    # keep a reference to all lines by keeping them in a list
    lines = list()

    # Style variables
    python_green = "#476042"
    point_size = 5
    line_width = 2

    # Create the window, a canvas, a grid and a rectangle with ply dimensions
    root = tk.Tk()
    c = tk.Canvas(root, width=COLS, height=ROWS, background='white')


    # Create background
    def create_background():

        x=[0,COLS,COLS,0,0]
        y=[0,0,ROWS,ROWS,0]
        for idx in range(len(x)-1):
            c.create_line(x[idx], y[idx], x[idx+1], y[idx+1], width=4, fill='red')

        width = COLS; height = ROWS
        for line in range(0, width, 10*scale): # range(start, stop, step)
            c.create_line([(line, 0), (line, height)], fill='black', tags='grid_line_w',width = 0.4)

        for line in range(0, height, 10*scale):
            c.create_line([(0, line), (width, line)], fill='black', tags='grid_line_h',width = 0.4)

    # Create mouse events
    def add_point(e):
       x, y = e.x, e.y
       c.pointsX.append(x/scale)
       c.pointsY.append(LY-y/scale)
       x1, y1 = (e.x - point_size/2), (e.y - point_size/2)
       x2, y2 = (e.x + point_size/2), (e.y + point_size/2)
       c.create_oval(x1, y1, x2, y2, fill=python_green)

       if c.old_coords:
          x1, y1 = c.old_coords
          c.create_line(x, y, x1, y1, width=line_width)
       c.old_coords = x, y

    def restart():
        """ Deletes all points and segments """
        c.delete("all")
        create_background()
        c.old_coords = None
        c.pointsX = list()
        c.pointsY = list()

    # Load an save
    def save_distribution():
        """ Save c.pointsX and c.pointsY in a txt file """
        # Open a pop-up window to select path and filename
        file_path = filedialog.asksaveasfilename(initialdir = os.path.join(self.path,self.folders['4_INFORMATION']),title = "Select file",filetypes = (("txt files","*.txt"),("all files","*.*")))
        # Save c.pointsX and c.pointsY in a txt file
        np.savetxt(file_path, (c.pointsX, c.pointsY), fmt='%d')
        self.INFO['fiber distribution filename'] = os.path.basename(file_path)

    def load_distribution():
        """ Load pointsX and pointsY from a txt file """
        # Delete all points and segments
        restart()
        # Open a pop-up window to select path and filename
        file_path = filedialog.askopenfilename(initialdir = os.path.join(self.path,self.folders['4_INFORMATION']),title = "Select file",filetypes = (("txt files","*.txt"),("all files","*.*")))
        self.INFO['fiber distribution filename'] = os.path.basename(file_path)
        # Load c.pointsX and c.pointsY from a txt file
        c.pointsX, c.pointsY = np.loadtxt(file_path, dtype=float)
        for idx in range(len(c.pointsX)):
            c.pointsX[idx] = c.pointsX[idx] * scale
            c.pointsY[idx] = (LY-c.pointsY[idx]) * scale 
        # Draw points
        for i in range(len(c.pointsX)):
            x = c.pointsX[i]
            y = c.pointsY[i]
            x1, y1 = (x - point_size/2), (y - point_size/2)
            x2, y2 = (x + point_size/2), (y + point_size/2)
            c.create_oval(x1, y1, x2, y2, fill=python_green)
        # Draw segments
        for i in range(len(c.pointsX)-1):
            x1 = c.pointsX[i]
            y1 = c.pointsY[i]
            x2 = c.pointsX[i+1]
            y2 = c.pointsY[i+1]
            c.create_line(x1, y1, x2, y2, width=line_width)


    restart()

    # Events

    c.pack()
    c.bind("<ButtonPress-1>", add_point)

    # Add mouse position in a label
    label = tk.Label(root)
    label.pack()
    c.bind("<Motion>", lambda event: label.configure(text=f"X = {event.x/scale} mm, Y = {LY-event.y/scale} mm"))


    # Create a top menu
    top = tk.Menu(root)
    root.config(menu=top)
    # Create a pulldown menu, and add it to the menu bar
    filemenu = tk.Menu(top, tearoff=0)
    filemenu.add_command(label="Open", command=load_distribution)
    filemenu.add_command(label="Save", command=save_distribution)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.quit)
    top.add_cascade(label="File", menu=filemenu)
    # Create another pulldown menu
    editmenu = tk.Menu(top, tearoff=0)
    editmenu.add_command(label="Erase all", command=restart)
    #editmenu.add_command(label="Cut", command=None)
    #editmenu.add_command(label="Copy", command=None)
    #editmenu.add_command(label="Paste", command=None)
    top.add_cascade(label="Edit", menu=editmenu)
    # Create another pulldown menu
    helpmenu = tk.Menu(top, tearoff=0)
    helpmenu.add_command(label="About", command=None)
    top.add_cascade(label="Help", menu=helpmenu)
    # Start the main loop

    root.mainloop()
