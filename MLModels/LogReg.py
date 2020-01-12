import numpy as np
import matplotlib.pyplot as plt

from models import Model
from neuron import Neuron
from function_dessection import *


outs = []
deltas = []
grads = []
xs = []
ys = []

def _shape(lst: List, *args)->np.ndarray:
    return np.array(lst).reshape(args)


class MlModel(Model):

    def __init__(self, activation_f, derivative_f, loss_f, inner_unit: Neuron):
        self.activation_f = activation_f
        self.derivative_f = derivative_f
        self.loss_f = loss_f
        self.inner_unit = inner_unit

    def fit(self, x: np.ndarray, y: np.ndarray):
        for xi, yi in zip(x, y):
            xim = _shape([xi], 1, 1)
            outcome = self.predict(xim)[0]
            delta = self.loss_f(yi, outcome)
            grad = self.derivative_f(outcome)

            ons = np.ones((xim.shape[0], 1))
            xim = np.concatenate((ons, xim), axis=1)
            adj = xim.transpose() * (delta)

            outs.append(outcome)
            deltas.append(adj[0].item())
            grads.append(grad)
            xs.append(xi)
            ys.append(yi)
            self.inner_unit.update(adj)
            print("Outcome:[{}]   Delta:[{}]     grad:[{}]".format(outcome, delta, adj))
            print(self.inner_unit)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.activation_f(xi) for xi in self.inner_unit.apply(x)])

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> str:
        return super().evaluate(x, y)


def test_mlmodel(x, y):
    np.random.seed(42)
    neur = Neuron(np.array(np.random.random(2) * 10).reshape(1, 2))
    model = MlModel(sigm, sigm_d, mse, neur)
    for i in range(10):
        model.fit(x, y)

    x = np.array(x).reshape(x.shape[0], 1)
    model.evaluate(x, y)

    for plotit, lbl in zip([outs, deltas, grads], ['outs', 'deltas', 'grads']):
        plt.plot(xs, plotit, 'o', label=lbl)

    post_plot(plt)


if __name__ == '__main__':
    import pandas as pd

    tsv = pd.read_csv('/tmp/transp.tsv', sep='\t', header=None)
    tsv = tsv.transpose()

    tsv = tsv.sample(frac=1)

    # plt.plot(tsv[0], tsv[1], 'ro')
    # post_plot(plt)

    succ = sum(tsv[tsv[0] <= 3][1])
    alls = len(tsv[tsv[0] <= 3])

    test_mlmodel(tsv[0], tsv[1])

# (1/m)*(
#         ((-y).T @ np.log(h + epsilon))
#             -((1-y).T @ np.log(1-h + epsilon)))
