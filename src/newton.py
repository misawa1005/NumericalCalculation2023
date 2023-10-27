import numpy as np
import matplotlib.pyplot as plt


def func_f(x):
    return x**2 - 10


def derivative(func_x, x, h=1e-10):
    """
    Calculate the derivative of a function
    :param func_x: function
    :param x: point
    :param h: step size
    :return: derivative
    """
    return (func_x(x + h) - func_x(x - h)) / (2 * h)


def newton(func_x, x0, eps=1e-10, error=1e-10, max_loop=100000):
    """
    Newton's method for finding roots of a function
    :param func_x: function
    :param x0: initial guess
    :param eps: tolerance for the root
    :param error: tolerance for the error
    :param max_loop: maximum number of iterations
    :return: root, number of iterations, error
    """
    x = x0
    for _ in range(max_loop):
        x1 = x - func_x(x) / derivative(func_x, x, eps)
        if abs(x1 - x) < eps or abs(func_x(x1)) < error:
            break
        x = x1
    return x


def visualization(func_x, x_min, x_max, x_solved):
    """
    Visualization of the function and the root
    :param func_x: function
    :param x_min: minimum x
    :param x_max: maximum x
    :param x_solved: root
    :return: None
    """
    exact_x = np.linspace(x_min, x_max, 500)
    exact_y = func_x(exact_x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.plot(exact_x, exact_y, "b")
    plt.scatter(x_solved, 0.0)
    plt.text(x_solved, 0.0, f"x = {str(x_solved)}", va="bottom", color="#0000ff")
    plt.xlim(x_min, x_max)
    plt.ylim(np.min(exact_y) - 5, np.max(exact_y) + 5)
    plt.savefig("newton.png")


if __name__ == "__main__":
    solution = newton(func_f, 2.0)

    visualization(func_f, solution - 1.0, solution + 1.0, solution)
