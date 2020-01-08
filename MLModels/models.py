import numpy as np


class Model:

    def fit(self, x: np.ndarray, y: np.ndarray)->str:
        raise NotImplemented()

    def predict(self, x: np.ndarray)->np.ndarray:
        raise NotImplemented()

    def evaluate(self, x: np.ndarray, y: np.ndarray)->str:
        raise NotImplemented()


class LinearModel(Model):

    def __init__(self, x: np.ndarray, b: np.ndarray):
        self.x = x
        self.b = b


def model_test():
    lm = LinearModel([4], [7])

    if not isinstance(lm, LinearModel):
        raise Exception(lm)


def included_test():
    model_test()


if __name__ == '__main__':
    included_test()
    print("Tests success!")
