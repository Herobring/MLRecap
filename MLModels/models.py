import numpy as np
import math


class Model:

    def fit(self, x: np.ndarray, y: np.ndarray)->str:
        raise NotImplemented()

    def predict(self, x: np.ndarray)->np.ndarray:
        raise NotImplemented()

    def evaluate(self, x: np.ndarray, y: np.ndarray)->str:
        raise NotImplemented()


class LinearReg(Model):

    def __init__(self, b: np.ndarray=None):
        self.b = b

    def fit(self, x: np.ndarray, y: np.ndarray)->str:
        ons = np.ones((x.shape[0], 1))
        x = np.concatenate((ons, x), axis=1)
        inv = np.linalg.inv(np.dot(np.transpose(x), x))
        residual_vec = np.dot(np.dot(inv, np.transpose(x)), y)
        self.b = residual_vec
        return 'b={}'.format(self.b)

    def predict(self, x: np.ndarray)->np.ndarray:
        ons = np.ones((x.shape[0], 1))
        x = np.concatenate((ons, x), axis=1)
        return np.dot(x, self.b)

    def evaluate(self, x: np.ndarray, y: np.ndarray)->str:
        return str(np.square(self.predict(x) - y).mean(axis=None))


def generate(input=range(-10, 10), func=None):
    eps = np.random.rand(len(input)) * 10

    def pol(x):
        cofs = [1, 0, -4]
        dem = len(cofs)
        res = 0
        for i in range(dem-1, -1, -1):
            res = res + math.pow(x, i)*cofs[dem-1-i]
        return res

    if not func:
        func = pol

    return np.add([func(x) for x in input], eps)


def model_test():
    lm = LinearReg([4])

    if not isinstance(lm, LinearReg):
        raise Exception(lm)


def included_test():
    model_test()


if __name__ == '__main__':
    included_test()
    print("Tests success!")
    print(generate())
    from visual import *

    x = np.array(list(range(-10, 10, 1))).reshape(20, 1)

    def lin_func(x):
        return -2*x + 10

    y = generate(np.ndarray.flatten(x), lin_func)
    d1_plot(x, y)
    lr = LinearReg()
    print(lr.fit(x, y))
    x2 = np.array(list(range(10, 30, 1))).reshape(20, 1)
    y2 = generate(np.ndarray.flatten(x2), lin_func)
    print(lr.evaluate(x2,y2))
    print(lr.predict(x2))
    plt.plot(list(x) + list(x2), list(y) + list(y2))
    plt.plot(np.append(x, x2), np.append(lr.predict(x), lr.predict(x2)))
    plt.show()

