import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go

from typing import Dict, Tuple, List


def funcs2graph(fs: np.array, interval=np.arange(-10, 10, 0.05)):

    start = min(interval)
    end = max(interval)
    plt.xlim(start*2, end*2)
    plt.ylim(start*2, end*2)

    # Don't mess with the limits!
    plt.autoscale(False)

    for f in fs:
        plt.plot(interval, [f(x) for x in interval], 'o', label=f.__name__)
    post_plot(plt)


def post_plot(plt):
    plt.grid()
    plt.legend()
    plt.show()


def idf(x):
    return x


def e_pow_x(x):
    return math.e ** x


def fun_gen(inner_f, interval):
    def f(x):
        return inner_f(x) / sum([inner_f(xi) for xi in interval])
    return f


def sigm(x):
    return 1 / (1 + math.exp(-x))


def sigm_d(x):
    return sigm(x) * (1 - sigm(x))


def sigm_dd(x):
    return sigm(x) * (1 - sigm(x)) * (1 - sigm(x)) + sigm(x) * (-1 * sigm(x) * (1 - sigm(x)))


def mse(x, y):
    return np.square(x-y)

if __name__ == '__main__':
    print('Start')
    # funcs2graph([idf, e_pow_x], np.arange(-2, 2, 0.05))
    x = np.arange(-6, 6, 0.1)
    # x = [1, 2, 3, 40, 41]
    y = [xi / sum(x) for xi in x]

    res_f = fun_gen(idf, x)
    res_f.__name__ = 'softmax_id'
    res_f2 = fun_gen(e_pow_x, x)
    print([sigm(xi) for xi in x])
    # funcs2graph([res_f, res_f2, e_pow_x, sigm], x)
    funcs2graph([sigm, sigm_d, sigm_dd], x)

