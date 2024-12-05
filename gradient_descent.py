import numpy as np
import matplotlib.pyplot as plt
import os 
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plot:
    def __init__(self, func):
        """Initialize with a callable function."""
        self.func = func

    def contour(self, x_range, y_range, resolution=100):
        """Plot a contour."""
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        plt.contour(X, Y, Z, levels=50, cmap='inferno')
        plt.colorbar()
        plt.show()

    def density_3d(self, x_range, y_range, resolution=100):
        """Plot a 3D density surface."""
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.view_init(elev=30, azim=120)
        plt.show()

class Rosenbrock:

    name = "Rosenbrock's function"
    equation = r"$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$"

    @staticmethod
    def function(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    @staticmethod
    def log_function(x, y):
        return np.log((1 - x)**2 + 100 * (y - x**2)**2)

    @staticmethod
    def gradient(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])

    @staticmethod
    def hessian(x, y):
        dxx = 2 - 400 * (y - 3 * x**2)
        dxy = -400 * x
        dyy = 200
        return np.array([[dxx, dxy], [dxy, dyy]])


class Himmelblau:

    name = "Himmelblau's function"
    equation = r"$f(x,y)=(x^2+y-11)^2 + (x + y^2 - 7)^2$"

    @staticmethod
    def function(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    @staticmethod
    def log_function(x,y):
        return np.log((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

    @staticmethod
    def gradient(x, y):
        dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        return np.array([dx, dy])

    @staticmethod
    def hessian(x, y):
        dxx = 12 * x**2 + 4 * y - 42
        dxy = 4 * x + 4 * y
        dyy = 12 * y**2 + 4 * x - 26
        return np.array([[dxx, dxy], [dxy, dyy]])


# example usage
Plot(Rosenbrock.function).density_3d(x_range=[-2,2],y_range=[-1,3])
plt.close()


