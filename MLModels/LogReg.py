from models import Model
from neuron import Neuron
from function_dessection import *


dzs = []
deltas = []
grads = []
xs = []
ys = []


def _shape(lst: List, *args)->np.ndarray:
    return np.array(lst).reshape(args)


class DenseLayer(Model):

    def __init__(self, activation_f, derivative_f, loss_f, input_size=1, output_size=1, lr=0.001):
        self.activation_f = activation_f
        self.derivative_f = derivative_f
        self.loss_f = loss_f
        self.weights = np.random.rand(output_size, input_size+1)
        self.lr = lr

    def fit(self, x: np.ndarray, y: np.ndarray):
        for xi, yi in zip(x, y):
            xim = _shape([xi], 1, len(xi))
            xim = self._b_add(xim)
            outcome = self._predict(xim)
            delta = self.loss_f(outcome, yi)
            grad = self.derivative_f(outcome)
            dz = delta @ grad
            dw = dz @ xim
            self.weights = self.weights - self.lr * dw
            dzs.append(dz.item(0))
            deltas.append(delta.item(0))
            grads.append(grad.item(0))
            xs.append(xi)
            ys.append(yi)

    def _b_add(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    def _predict(self, xb: np.ndarray) -> np.ndarray:
        nn_out = self.weights @ xb.transpose()
        return self.activation_f(nn_out)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._predict(self._b_add(x))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> str:
        return super().evaluate(x, y)


def test_mlmodel(x, y):
    np.random.seed(42)
    model = DenseLayer(sigm, sigm_d, mae, 1, 1, lr=0.1)
    for i in range(1000):
        model.fit(x, y)

    print(model.weights)
    x = np.array(x).reshape(x.shape[0], 1)
    model.evaluate(x, y)

    for xx, yy, pp in zip(x, y, model.predict(x)[0]):
        print("xx={} yy={} pp={}".format(xx, yy, pp))

    for i, (plotit, lbl) in enumerate(zip([dzs, deltas, grads], ['dz', 'deltas', 'grads'])):
        plt.plot(range(len(plotit)), plotit, 'o', label=lbl)

    post_plot(plt)


if __name__ == '__main__':
    import pandas as pd

    tsv = pd.read_csv('/tmp/transp.tsv', sep='\t', header=None)  # 1.5046   -4
    tsv = tsv.transpose()

    tsv = tsv.sample(frac=1)

    succ = sum(tsv[tsv[0] <= 3][1])
    alls = len(tsv[tsv[0] <= 3])

    size = len(tsv)
    x = _shape(tsv[0], size, 1)
    y = _shape(tsv[1], size, 1)
    test_mlmodel(x, y)
