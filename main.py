import numpy as np
import matplotlib.pyplot as plt


def newton(F, dF, x0, delta=1e-04, ep=1e-04, maxIter=100):
    x_k = x0

    for i in range(maxIter):
        x = x_k - np.linalg.inv(dF(x_k)) @ F(x_k)
        if np.linalg.norm(F(x_k)) < ep:
            break
        if np.linalg.norm(x - x_k) < delta:
            break
        x_k = x

    return x_k


def newton_unity_root(z0s, d, delta=1e-04, ep=1e-04, maxIter=15):
    """
    Same as `newton`, but optimized for `z**d = 1` functions.

    Calculating the fractals would work with `newton` without having to
    make any adjustments to the function, but it takes forever as we
    loop over every single point.

    Here we have vectorized this process and only loop over each iteration.
    """
    z_k = z0s

    for i in range(maxIter):
        a = (d - 1) * z_k
        b = 1 / np.power(z_k, d - 1)
        z = (a + b) / d
        if np.linalg.norm(np.power(z_k, d) - 1) < ep:
            break
        if np.linalg.norm(z - z_k) < delta:
            break
        z_k = z

    return z_k


def colorize_points(points_n, color_points):
    points = np.vstack([points_n for _ in range(len(color_points))]).T
    dif = np.abs(points - color_points)
    closest_index = np.argmin(dif, axis=1)
    return closest_index


def plot_simple_function():
    """
    Calculate roots for f: R -> R with newtons method and plot it.
    """

    def F(x):
        x = np.array(x)
        y = np.array(x**3 - 2 * x).reshape(1, 1)
        return y

    def dF(x):
        x = np.array(x)
        y = np.array(3 * x**2 - 2).reshape(1, 1)
        return y

    x0s = [
        np.array(0.1).reshape(
            1,
        ),
        np.array(2).reshape(
            1,
        ),
        np.array(-2).reshape(
            1,
        ),
    ]
    delta = 1e-10
    ep = 1e-10
    maxIter = 50

    x0_values = np.array([newton(F, dF, x0, delta, ep, maxIter) for x0 in x0s])
    y0_values = np.zeros(x0_values.shape[0])
    x_values = np.linspace(-2, 2, 100)
    y_values = np.squeeze(np.array([F(x) for x in x_values]))

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.scatter(x0_values, y0_values, color="red", linewidth=10)
    plt.plot(x_values, y_values, linewidth=3)

    plt.title("x -> x**3 - 2*x")
    plt.tight_layout()
    plt.grid()
    plt.show()


def calculate_second_function():
    """
    Calculate root of f: R**2 -> R**2 and print it.
    """

    def F(x):
        x = np.array(x)
        y = np.array((x[0] ** 2 + x[1] ** 2 - 6 * x[0], 3 / 4 * np.exp(-x[0]) - x[1]))
        return y

    def dF(x):
        x = np.array(x)
        y = np.array((2 * x[0] - 6, 2 * x[1], -3 / 4 * np.exp(-x[0]), -1)).reshape(2, 2)
        return y

    x0 = np.array((2 / 25, 7 / 10))
    delta = 1e-10
    ep = 1e-10
    maxIter = 50

    x0_value = newton(F, dF, x0, delta, ep, maxIter)
    print(f"(Task 3) Root of F: R**2 -> R**2 = {x0_value}")


def plot_simple_fractal():
    """
    Colorize the closest root to a tight point grid to visualize fractals.

    Uses `z**3 = 1` function.
    """
    steps = 512
    x0s = np.linspace(-1, 1, steps)
    y0s = np.linspace(-1, 1, steps)
    xx, yy = np.meshgrid(x0s, y0s)
    grid = np.ravel(xx + 1j * yy)

    delta = 1e-05
    ep = 1e-05
    maxIter = 15
    result = newton_unity_root(grid, 3, delta, ep, maxIter)

    # real_x0s = np.array([np.exp(2j * np.pi * k / 3) for k in range(3)])
    real_x0s = np.array((1, -0.5 + 0.5j * np.sqrt(3), -0.5 - 0.5j * np.sqrt(3)))
    colored_x0s = colorize_points(result, real_x0s)

    plt.imshow(colored_x0s.reshape(steps, steps))
    plt.title("Colored lattice points based on nearest location to root of z**3 = 1")
    plt.show()


def plot_trippy_fractal():
    """
    Colorize the closest root to a tight point grid to visualize fractals.

    Uses `z**5 = 1` function.
    """

    # def F(z):
    #     z = np.array(z)
    #     y = np.array(np.power(z, 5) - 1).reshape(1, 1)
    #     return y
    #
    # def dF(z):
    #     z = np.array(z)
    #     y = np.array(5 * np.power(z, 4)).reshape(1, 1)
    #     return y
    #
    steps = 512
    x0s = np.linspace(-1, 1, steps)
    y0s = np.linspace(-1, 1, steps)
    xx, yy = np.meshgrid(x0s, y0s)
    grid = np.ravel(xx + 1j * yy)

    delta = 1e-14
    ep = 1e-14
    maxIter = 15
    result = newton_unity_root(grid, 5, delta, ep, maxIter)
    # result = np.array([newton(F, dF, x0, delta, ep, maxIter) for x0 in grid])

    plt.imshow(np.angle(result.reshape(steps, steps)), cmap="hsv")
    plt.title("z**5 = 1")
    plt.show()


def main():
    # plot_simple_function()
    # calculate_second_function()
    # plot_simple_fractal()
    plot_trippy_fractal()


if __name__ == "__main__":
    main()
