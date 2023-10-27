import pandas as pd
import numpy as np


def f(x):
    return x + np.sin(x) - 2


def squeeze(a, b, ex=1e-6):
    n = 1
    error_a = 1
    error_b = 1

    x_i_list = []
    y_i_list = []

    while error_a > ex and error_b > ex:
        x_i = (a * f(b) - b * f(a)) / (f(b) - f(a))
        y_i = f(x_i)

        x_i_list.append(x_i)
        y_i_list.append(y_i)

        posi_nega = y_i * f(b)

        error_a = abs(x_i - a) / abs(a)
        error_b = abs(x_i - b) / abs(b)

        if posi_nega < 0:
            a = x_i
        else:
            b = x_i
        n += 1

    df = pd.DataFrame({"n": list(range(1, n)), "x_i": x_i_list, "y_i": y_i_list})
    return df


if __name__ == "__main__":
    print("*** はさみうち法を用いて解を求める ***")
    print("収束判定の値を代入してください")
    # ex = float(input("ex = "))
    print("探索範囲となる2点a, bを入力してください")
    a = float(input("a = "))
    b = float(input("b = "))

    print("解は以下の通りです")
    df = squeeze(a, b)
    print(df)
