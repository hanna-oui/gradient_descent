import numpy as np
import matplotlib.pyplot as plt

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
    name = "Rosenbrock"
    equation = r"$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$"

    @staticmethod
    def fn(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    @staticmethod
    def log_fn(x, y):
        return np.log((1 - x)**2 + 100 * (y - x**2)**2)
    
    @staticmethod 
    def function(X):
        return (1 - X[0])**2 + 100 * (X[1] - X[0]**2)**2

    @staticmethod
    def gradient(X):
        dx = -2 * (1 - X[0]) - 400 * X[0] * (X[1] - X[0]**2)
        dy = 200 * (X[1] - X[0]**2)
        return np.array([dx, dy])

    @staticmethod
    def hessian(X):
        dxx = 2 - 400 * (X[1] - 3 * X[1]**2)
        dxy = -400 * X[0]
        dyy = 200
        return np.array([[dxx, dxy], [dxy, dyy]])


class Himmelblau:
    name = "Himmelblau"
    equation = r"$f(x,y)=(x^2+y-11)^2 + (x + y^2 - 7)^2$"

    @staticmethod
    def fn(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    @staticmethod
    def log_fn(x,y):
        return np.log((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

    @staticmethod
    def function(X):
        return (X[0]**2 + X[1] - 11)**2 + (X[0] + X[1]**2 - 7)**2

    @staticmethod
    def gradient(X):
        dx = 4 * X[0] * (X[0]**2 + X[1] - 11) + 2 * (X[0] + X[1]**2 - 7)
        dy = 2 * (X[0]**2 + X[1] - 11) + 4 * X[1] * (X[0] + X[1]**2 - 7)
        return np.array([dx, dy])

    @staticmethod
    def hessian(X):
        dxx = 12 * X[0]**2 + 4 * X[1] - 42
        dxy = 4 * X[0] + 4 * X[1]
        dyy = 12 * X[1]**2 + 4 * X[0] - 26
        return np.array([[dxx, dxy], [dxy, dyy]])
    

class GradientDescent:
    function_mapping = {
        "Rosenbrock" : Rosenbrock,  
        "Himmelblau" : Himmelblau
    }

    def __init__(self, function_name):
        function_class = self.function_mapping[function_name]
        self.f = function_class.function
        self.grad = function_class.gradient 
    
    def armijo_step(self, xk, beta=0.5, delta=0.1):
        tau = 1.0  # Initial step size
        gradient = self.grad(xk)
        while self.f(xk - tau * gradient) > self.f(xk) - delta * tau * np.dot(gradient, gradient):
            tau *= beta
        return tau
    
    def optimize(self, x0, step_type, fixed_tau=0.001, epsilon=10e-5):
        xk = x0 
        i = 0
        while np.linalg.norm(self.grad(xk)) > epsilon:
            if step_type == "armijo":
                tau = self.armijo_step(xk)
            if step_type == "fixed":
                tau = fixed_tau
            xk = xk - tau*self.grad(xk) 
            i+=1
            print(f"Iteration: {i}")
            print(f"Gradient Norm: {np.linalg.norm(self.grad(xk))}")
            print(f"Function Value: {self.f(xk)}")
            print(f"Current point {xk}")
        return xk, i
            

print(GradientDescent("Himmelblau").optimize([np.pi+1,np.pi-1], "fixed"))