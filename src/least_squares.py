import numpy as np
import matplotlib.pyplot as plt


def least_squares(x, y):
    n = len(x)
    a = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x**2) - (sum(x)) ** 2)
    b = (sum(x**2) * sum(y) - sum(x) * sum(x * y)) / (n * sum(x**2) - (sum(x)) ** 2)
    return a, b


def plot_least_squares(x, y, a, b):
    plt.scatter(x, y, label="data")
    plt.plot(x, a * x + b, label="y=ax+b")
    plt.legend()
    plt.savefig("least_squares.png")


if __name__ == "__main__":
    x = np.array([1, 3, 7])
    y = np.array([2, 5, 7])
    a, b = least_squares(x, y)
    plot_least_squares(x, y, a, b)
