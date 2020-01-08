import matplotlib.pyplot as plt
import numpy as np


DEF_PATH = 'plot.png'


def d1_plot(x, y, path=DEF_PATH):
    print(x)
    print(y)
    plt.plot(x, y)
    plt.show()
    plt.savefig(path)
