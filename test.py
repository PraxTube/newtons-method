import numpy as np

from main import colorize_points


steps = 2
x0s = np.linspace(-1, 1, steps)
y0s = np.linspace(-1, 1, steps)
grid = np.array([[x + 1j * y for x in x0s] for y in y0s]).flatten()

n = 5
points_n = 2.0 * (np.random.rand(n) - 0.5)

points_3 = np.array([-1.0 + 0j, 0.5 + 0.5j, 0.8 - 0.8j])


print(grid)
print(colorize_points(grid, points_3))
