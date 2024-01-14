import numpy as np
import matplotlib.pyplot as plt


def newton(F, dF, x0, delta=1e-04, ep=1e-04, maxIter=100):
    x_k = x0
    for i in range(maxIter):
        x = x_k - F(x_k) / dF(x_k)
        if np.linalg.norm(F(x_k)) < ep:
            break
        if np.linalg.norm(x - x_k) < delta:
            break
        x_k = x

    return x_k


def plot_simple_function():
    """
    Calculate x_0 for f: R -> R with newtons method
    """
    F = lambda x: x**3 - 2 * x
    dF = lambda x: 3 * x**2 - 2
    x0s = (0.1, 2, -2)
    delta = 1e-10
    ep = 1e-10
    maxIter = 50

    x0_values = np.array([newton(F, dF, x0, delta, ep, maxIter) for x0 in x0s])
    y0_values = np.zeros(x0_values.shape[0])
    x_values = np.linspace(-2, 2, 100)
    y_values = np.array([F(x) for x in x_values])

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.scatter(x0_values, y0_values, color="red", linewidth=10)
    plt.plot(x_values, y_values, linewidth=3)

    plt.title("x -> x**3 - 2*x")
    plt.tight_layout()
    plt.grid()
    plt.show()


def main():
    plot_simple_function()


if __name__ == "__main__":
    main()
