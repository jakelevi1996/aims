import torch

class Sgd:
    def __init__(self, model, learning_rate=1e-3):
        self._params = model.get_params()
        self._learning_rate = learning_rate

    def step(self):
        for param in self._params:
            param.data -= self._learning_rate * param.grad

class SgdMomentum:
    def __init__(self, model, momentum=0.3, learning_rate=1e-3):
        self._params = model.get_params()
        self._momentum = momentum
        self._learning_rate = learning_rate
        self._v = [torch.zeros_like(param) for param in self._params]

    def step(self):
        for v, param in zip(self._v, self._params):
            v *= self._momentum
            v += param.grad
            param.data -= self._learning_rate * v
