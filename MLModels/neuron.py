import numpy as np


class Neuron:

    def __init__(self, weights: np.ndarray):
        self.weights = weights.transpose()

    @staticmethod
    def _ones(x: np.ndarray)->np.ndarray:
        ones = np.ones((x.shape[0], 1))
        return np.hstack([ones, x])

    def apply(self, x: np.ndarray):
        return np.dot(Neuron._ones(x), self.weights)

    def update(self, weights: np.ndarray):
        self.weights = np.add(self.weights, weights)

    def __str__(self):
        return "{{Neuron:{}}}".format(self.weights)


def base_test():
    p = Neuron([1, 2, 3], 3)
    if not isinstance(p, Neuron):
        Exception('Created not a neuron')


def apply_test():
    p = Neuron([1, -1, 2, 0, 0], 100000)
    res = p.apply([5, 10, 5, 343, 34343])

    if res != 100000+5:
        raise Exception(res)


def update_test():
    k = [1, -1, 2, 0, 0]
    p = Neuron(k, 100000)
    p.update([5, 10, 5, 0, 1], -99999)

    if sum(np.equal(k, p.weights)) > 1:
        raise Exception(p.weights)

    if sum(np.not_equal(p.weights, [6, 9, 7, 0, 1])) > 0:
        raise Exception(p.weights)

    if p.b != 1:
        raise Exception(p.b)


def str_test():
    p = Neuron([1, -1, 2, 0, 0], 100000)
    if str(p) != '{Neuron:[1, -1, 2, 0, 0]+100000}':
        raise Exception(p)


def included_test():
    base_test()
    apply_test()
    update_test()
    str_test()


if __name__ == '__main__':
    included_test()
    print("Tests success!")
