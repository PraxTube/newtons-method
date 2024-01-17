"""
Reference videos on this topic to better understand what is happening:

https://www.youtube.com/watch?v=-RdOwhmqP5s
https://www.youtube.com/watch?v=LqbZpur38nw
"""

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


def mandelbrot_sequence(c, maxIter=256):
    z_k = c
    m = np.zeros(z_k.shape)

    for i in range(maxIter):
        z_k = np.power(z_k, 2) + c
        m += np.array(z_k * z_k <= 4, dtype=float)

    m /= maxIter
    return m
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
    steps = 512
    x0s = np.linspace(-1, 1, steps)
    y0s = np.linspace(-1, 1, steps)
    xx, yy = np.meshgrid(x0s, y0s)
    grid = np.ravel(xx + 1j * yy)

    delta = 1e-14
    ep = 1e-14
    maxIter = 15
    result = newton_unity_root(grid, 5, delta, ep, maxIter)

    plt.imshow(np.angle(result.reshape(steps, steps)), cmap="hsv")
    plt.title("z**5 = 1")
    plt.show()


def plot_minima_function():
    def F(x1, x2):
        return (x1 + 1) ** 4 + (x2 - 1) ** 4

    # Jacobi Matrix of F
    def J(x):
        x = np.array(x)
        y = np.array([4 * np.power(x[0] + 1, 3), 4 * np.power(x[1] - 1, 3)])
        return y

    # Hessian Matrix of F
    def H(x):
        x = np.array(x)
        y = np.array(
            [12 * np.power(x[0] + 1, 2), 0, 0, 12 * np.power(x[1] - 1, 2)]
        ).reshape(2, 2)
        return y

    x0 = np.array((-1.1, 1.1))
    delta = 1e-15
    ep = 1e-15
    maxIter = 150

    x0_value = newton(J, H, x0, delta, ep, maxIter)

    x1 = np.linspace(-5, 3, 10)
    x2 = np.linspace(-3, 5, 10)
    X1, X2 = np.meshgrid(x1, x2)
    Z = F(X1, X2)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(X1, X2, Z, rstride=1, cstride=1, label="F: R**2 -> R")
    ax.scatter(
        -1,
        1,
        F(x0_value[0], x0_value[1]),
        color="red",
        marker="o",
        s=200,
        label="Minimum",
    )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("F")
    ax.legend()

    plt.title(f"Minimum at {x0_value}")
    plt.show()


def plot_mandelbrot():
    steps = 1024
    x0s = np.linspace(-1.5, 0.5, steps)
    y0s = np.linspace(-1, 1, steps)
    xx, yy = np.meshgrid(x0s, y0s)
    grid = np.ravel(xx + 1j * yy)

    maxIter = 256
    result = mandelbrot_sequence(grid, maxIter)

    plt.axis("off")
    plt.imshow(result.reshape(steps, steps), cmap="inferno")
    plt.show()


def main():
    # plot_simple_function()
    # calculate_second_function()
    # plot_simple_fractal()
    # plot_trippy_fractal()
    # plot_minima_function()
    plot_mandelbrot()


if __name__ == "__main__":
    main()
