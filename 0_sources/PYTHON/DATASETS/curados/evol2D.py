import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def distribution2map(self):

    """ Function get mean Temperature and mean and maximum Deformation along a curing process """

    # Get fiber distribution from file
    fiber_distribution_path = os.path.join(self.path,self.folders['4_INFORMATION'],self.INFO['fiber distribution filename'])

    if not os.path.exists(fiber_distribution_path):
        print('\nNo fiber distribution found in:',fiber_distribution_path)
        print('Please run:')
        print('     DATASET.fiber_distribution()')
        print('to draw it and save it')

    pointsX, pointsY = np.loadtxt(fiber_distribution_path, dtype=float)

    """
    # Divide line drawn by (pointsX and pointsY) to have more points along the same spline
    nb_points = len(pointsX)
    alpha = 20
    pointsX_new = np.zeros(nb_points*alpha)
    pointsY_new = np.zeros(nb_points*alpha)
    for i in range(nb_points-1):
        pointsX_new[i*alpha:(i+1)*alpha] = np.linspace(pointsX[i],pointsX[i+1],alpha)
        pointsY_new[i*alpha:(i+1)*alpha] = np.linspace(pointsY[i],pointsY[i+1],alpha)
    """

    L = list()
    for x,y in zip(pointsX,pointsY):
        L.append((x,y))

    X = np.array(L).T

    t = np.linspace(0,1,len(pointsX))
    t_new = np.linspace(0,1,10*len(pointsX))
    f = interp1d(t, X, kind='quadratic')

    pointsX_new = list()
    pointsY_new = list()

    for pos in t_new:
        x,y = f(pos)
        pointsX_new.append(x)
        pointsY_new.append(y)

    # Smooth the polygonal line by adding fillets
    min_radius = 1 # mm
    def add_fillets(x,y,r):
        """ Function to add fillets to a polygonal line defined by x and y coordinates of the vertices
            by adding some points in the corner """
        x_new = []
        y_new = []
        for i in range(len(x)-1):
            x_new.append(x[i])
            y_new.append(y[i])
            if np.sqrt((x[i]-x[i+1])**2+(y[i]-y[i+1])**2) > r:
                n = int(np.ceil(np.sqrt((x[i]-x[i+1])**2+(y[i]-y[i+1])**2)/r))
                for j in range(n):
                    x_new.append(x[i]+(x[i+1]-x[i])/(n+1)*(j+1))
                    y_new.append(y[i]+(y[i+1]-y[i])/(n+1)*(j+1))
        x_new.append(x[-1])
        y_new.append(y[-1])
        return x_new, y_new

    pointsX, pointsY = add_fillets(pointsX,pointsY,min_radius)

    # Add fillets to the polygonal line
    pointsX_new,pointsY_new = add_fillets(pointsX,pointsY,min_radius)

    plt.figure()
    plt.plot(pointsX, pointsY, 's-')
    plt.plot(pointsX_new, pointsY_new, 'o--')
    #plt.plot(f(t_new), 'o--')
    plt.grid()
    plt.show()
