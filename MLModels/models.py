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


class NonLinearReg(Model):

    def __init__(self, power):
        self.power = power
        self.b = None

    def _pol_x(self, x: np.ndarray)->np.ndarray:
        x_pol = x
        for i in range(1, self.power):
            x_pol = np.concatenate((x_pol, np.power(x, i+1)), axis=1)
        ons = np.ones((x_pol.shape[0], 1))
        x_pol = np.concatenate((ons, x_pol), axis=1)
        return x_pol

    def fit(self, x: np.ndarray, y: np.ndarray)->str:
        x = self._pol_x(x)
        inv = np.linalg.inv(np.dot(np.transpose(x), x))
        residual_vec = np.dot(np.dot(inv, np.transpose(x)), y)
        self.b = residual_vec
        return 'b={}'.format(self.b)

    def predict(self, x: np.ndarray)->np.ndarray:
        x = self._pol_x(x)
        return np.dot(x, self.b)

    def evaluate(self, x: np.ndarray, y: np.ndarray)->str:
        return str(np.square(self.predict(x) - y).mean(axis=None))


def generate(input=range(-10, 10), func=None):
    input = np.array(input)
    sz = len(input)
    # het = [0] * (sz//2) + [1] * round(sz/2)
    # input = input* het

    def pol(x):
        cofs = [1, 0, -4]
        dem = len(cofs)
        res = 0
        for i in range(dem-1, -1, -1):
            res = res + math.pow(x, i)*cofs[dem-1-i]
        return res

    if not func:
        func = pol

    x = [func(x) for x in input]
    mn = np.mean(x)
    eps = np.random.rand(sz) * mn - mn//2


    return np.add(x, eps)


def model_test():
    lm = LinearReg([4])

    if not isinstance(lm, LinearReg):
        raise Exception(lm)


def non_lin_model_test():
    lm = NonLinearReg(4)

    if not isinstance(lm, NonLinearReg):
        raise Exception(lm)


def included_test():
    model_test()
    non_lin_model_test()


if __name__ == '__main__':
    included_test()
    print("Tests success!")
    from visual import *

    def func(x):
        return +6*x + 10 + 38*x*x - x*x*x

    x = np.array(list(range(-10, 30, 1))).reshape(40, 1)
    y = generate(np.ndarray.flatten(x), func)

    print("X={}".format(x))
    print("Y={}".format(y))
    d1_plot(x, y)
    if False:
        lr = LinearReg()
        print(lr.fit(x, y))
        x2 = np.array(list(range(10, 30, 1))).reshape(20, 1)
        y2 = generate(np.ndarray.flatten(x2), func)
        print(lr.evaluate(x2,y2))

    if True:
        lr = NonLinearReg(6)
        print(lr.fit(x, y))
        x2 = np.array(list(range(10, 30, 1))).reshape(20, 1)
        y2 = generate(np.ndarray.flatten(x2), func)
        print(lr.evaluate(x2,y2))

    # plt.plot(list(x) + list(x2), list(y) + list(y2), 'ro')
    # plt.plot(np.append(x, x2), np.append(lr.predict(x), lr.predict(x2)))
    plt.plot(list(x), list(y), 'ro')
    plt.plot(x, lr.predict(x))

    plt.show()

