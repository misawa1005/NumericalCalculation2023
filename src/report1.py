import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from newton import newton
from dataclasses import dataclass
from pathlib import Path


@dataclass
class coef:
    a: float
    b: float
    c: float
    d: float


def optimal_clusters_elbow_method(data: np.ndarray, max_clusters: int = 10):
    """
    エルボー法を用いて最適なクラスタ数を見つける
    :param data: クラスタリングするデータ
    :param max_clusters: 試行する最大のクラスタ数
    :return: 最適なクラスタ数
    """
    distortions = []
    for n in range(1, max_clusters):
        kmeans = KMeans(n_clusters=n, n_init=10)
        kmeans.fit(data.reshape(-1, 1))
        distortions.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, max_clusters), distortions, marker="o")
    plt.title("The Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion")
    plt.savefig("output/elbow_method.png")
    plt.close()
    delta_distortions = np.diff(distortions) / distortions[:-1]

    optimal_clusters = np.argmin(delta_distortions) + 2
    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10)
    kmeans.fit(data.reshape(-1, 1))

    labels = kmeans.labels_

    cluster_averages = []
    for i in range(optimal_clusters):
        cluster_data = data[labels == i]
        cluster_average = np.mean(cluster_data)
        cluster_averages.append(cluster_average)

    return cluster_averages


def least_squares(x: np.ndarray, y: np.ndarray) -> coef:
    """
    最小二乗法を用いてデータを3次式で近似する

    """

    x_powers = [np.sum(x**i) for i in range(7)]

    xy_powers = [np.sum((x**i) * y) for i in range(4)]

    A = np.array(
        [
            [x_powers[3], x_powers[2], x_powers[1], x_powers[0]],
            [x_powers[4], x_powers[3], x_powers[2], x_powers[1]],
            [x_powers[5], x_powers[4], x_powers[3], x_powers[2]],
            [x_powers[6], x_powers[5], x_powers[4], x_powers[3]],
        ]
    )

    B = np.array(xy_powers)
    a, b, c, d = np.linalg.solve(A, B)
    return coef(a, b, c, d)


def plot_least_squares(x: np.ndarray, y: np.ndarray, func_x):
    plt.scatter(x, y, label="data")
    plt.plot(x, func_x(x))
    derivative_x = derivative_function(func_x)
    solutions = []
    for i in np.random.randint(0, len(x), 20).tolist():
        solutions.append(newton(derivative_x, x[i], 1e-5, 1e-5, 10000))
    solutions = np.array(solutions)
    cluster_averages = optimal_clusters_elbow_method(solutions)

    for solution in cluster_averages:
        plt.scatter(solution, func_x(solution))
        plt.text(solution, 0.0, f"x = {str(solution)}", va="bottom", color="#0000ff")
        plt.legend()
    plt.savefig("output/least_squares.png")


def derivative_function(func, h=1e-5):
    """
    与えられた関数の導関数を表す新しい関数を生成する
    :param func: 元の関数
    :param h: 微小な変化
    :return: 導関数を表す関数
    """

    def derivative(x):
        return (func(x + h) - func(x - h)) / (2 * h)

    return derivative


if __name__ == "__main__":
    data = np.loadtxt("data/data1024.csv", delimiter=",", skiprows=1)
    x = data[..., 0]
    y = data[..., 1]
    Path("output").mkdir(exist_ok=True, parents=True)
    coef = least_squares(x, y)
    func_x = lambda x: coef.a * x**3 + coef.b * x**2 + coef.c * x + coef.d
    plot_least_squares(x, y, func_x)
