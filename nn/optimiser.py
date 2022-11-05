class Sgd:
    def __init__(self, model, learning_rate=1e-3):
        self._params = model.get_params()
        self._learning_rate = learning_rate

    def step(self):
        for param in self._params:
            param.data -= self._learning_rate * param.grad
