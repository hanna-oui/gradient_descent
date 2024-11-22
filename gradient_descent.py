import numpy as np
import matplotlib.pyplot as plt
import os 

# Rosenbrock's function
def Rosenbrock_fn(x1,x2):
    return (1-x1)**2 + 100*(x2-x1**2)**2

# Himmelblau's function
def Himmelblau_fn(x1,x2):
    return (x1**2+x2-11)**2 + (x1+x2**2-7)**2

# Define plotting parameters
functions = [
    (Rosenbrock_fn, "Rosenbrock's function", 'viridis'),
    (Himmelblau_fn, "Himmelblau's function", 'plasma')
]

# Generate grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Create figure
fig = plt.figure(figsize=(12, 6))

# Loop through functions and plot
for i, (func, title, cmap) in enumerate(functions, start=1):
    ax = fig.add_subplot(1, 2, i, projection='3d')
    Z = func(X, Y)
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='k')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)


# Get the current directory and define the relative output path
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)  
filepath = os.path.join(output_dir, "plot3D")

# Save the plot
plt.savefig(filepath, format="png", dpi=300)
plt.show()  # Display the plot
plt.close()  # Close the figure to avoid display