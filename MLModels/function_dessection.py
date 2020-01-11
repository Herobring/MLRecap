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
        plt.plot(interval, [f(x) for x in interval], 'ro', label=f.__name__)

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


if __name__ == '__main__':
    print('Start')
    # funcs2graph([idf, e_pow_x], np.arange(-2, 2, 0.05))
    x = np.arange(-5, 67, 10)
    x = [1, 2, 3, 40, 41]
    y = [xi / sum(x) for xi in x]

    res_f = fun_gen(idf, x)
    res_f2 = fun_gen(e_pow_x, x)
    print([res_f(xi) for xi in x])
    print([res_f2(xi) for xi in x])
    funcs2graph([res_f, res_f2, e_pow_x], x)

