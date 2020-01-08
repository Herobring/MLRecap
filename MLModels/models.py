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

    def __init__(self, x: np.ndarray, b: np.ndarray):
        self.x = x
        self.b = b


def generate():
    eps = np.random.rand(20) * 10

    def pol(cofs, x):
        dem = len(cofs)
        res = 0
        for i in range(dem-1, -1, -1):
            res = res + math.pow(x, i)*cofs[dem-1-i]
        return res

    return np.add([pol([1, 0, -4], x) for x in range(-10, 10)], eps)


def model_test():
    lm = LinearReg([4], [7])

    if not isinstance(lm, LinearReg):
        raise Exception(lm)


def included_test():
    model_test()


if __name__ == '__main__':
    included_test()
    print("Tests success!")
    print(generate())
