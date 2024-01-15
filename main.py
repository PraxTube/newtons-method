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


def colorize_points(points_n, color_points):
    points = np.vstack([points_n for _ in range(len(color_points))]).T
    dif = points - color_points
    closest_index = np.argmin(dif * dif, axis=1)
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
    def F(z):
        z = np.array(z)
        y = np.array(z**3 - 1).reshape(1, 1)
        return y

    def dF(z):
        z = np.array(z)
        y = np.array(3 * z**2).reshape(1, 1)
        return y

    steps = 128
    x0s = np.linspace(-1, 1, steps)
    y0s = np.linspace(-1, 1, steps)
    grid = np.array([[x + 1j * y for x in x0s] for y in y0s]).flatten()

    delta = 1e-05
    ep = 1e-05
    maxIter = 15
    result = np.array([newton(F, dF, z0, delta, ep, maxIter) for z0 in grid]).reshape(-3)

    print(result.shape)

    real_x0s = np.array((-1, (-1 + 1j * np.sqrt(3)) / 2, (-1 - 1j * np.sqrt(3)) / 2))
    colored_x0s = colorize_points(result, real_x0s)

    plt.imshow(colored_x0s.reshape(steps, steps), cmap='hsv', extent=(-1, 1, -1, 1))

    # Plotting
    # plt.figure(figsize=(14, 6))
    # plt.scatter(grid, colored_x0s, linewidth=10)
    #
    # plt.title("x -> x**3 - 2*x")
    # plt.tight_layout()
    # plt.grid()
    plt.show()


def main():
    # plot_simple_function()
    # calculate_second_function()
    plot_simple_fractal()


if __name__ == "__main__":
    main()
