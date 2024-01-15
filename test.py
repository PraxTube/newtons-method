import numpy as np

from main import colorize_points


n = 5  # Replace with the actual size of your (n,) array
points_n = 2.0 * (np.random.rand(n) - 0.5)

points_3 = np.array([-1., 0.5, 0.8])  # Replace with your actual (3,) array

print(points_n[:, np.newaxis].shape)
print(points_n[:, np.newaxis])

print(colorize_points(points_n, points_3))
