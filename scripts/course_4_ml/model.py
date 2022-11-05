import torch
import torchvision
import numpy as np
import __init__
import plotting
import util

class Mlp:
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_hidden_layers,
        output_act,
        hidden_act,
        bias_std=0,
        rng=None,
    ):
        if rng is None:
            rng = np.random.default_rng(0)
        if num_hidden_layers > 0:
            self._layers = [
                Layer(input_dim, hidden_dim, hidden_act, bias_std, rng)
            ]
            for _ in range(num_hidden_layers - 1):
                self._layers.append(
                    Layer(hidden_dim, hidden_dim, hidden_act, bias_std, rng)
                )
            self._layers.append(
                Layer(hidden_dim, output_dim, output_act, bias_std, rng)
            )
        else:
            self._layers = [
                Layer(input_dim, output_dim, output_act, bias_std, rng)
            ]

    def forward(self, x):
        layer_output = x.T
        for layer in self._layers:
            layer_output = layer.forward(layer_output)
        return layer_output.T

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

    def get_params(self):
        param_list = [
            param
            for layer in self._layers
            for param in layer.get_params()
        ]
        return param_list

class Layer:
    def __init__(self, input_dim, output_dim, activation_func, bias_std, rng):
        w_std = np.sqrt(2 / input_dim)
        w0 = rng.normal(size=[output_dim, input_dim ]) * w_std
        b0 = rng.normal(size=[output_dim, 1         ]) * bias_std
        self._weights = torch.tensor(
            data=w0,
            dtype=torch.float32,
            requires_grad=True,
        )
        self._bias = torch.tensor(
            data=b0,
            dtype=torch.float32,
            requires_grad=True,
        )
        self._act_func = activation_func

    def forward(self, x):
        return self._act_func(self._weights @ x + self._bias)

    def zero_grad(self):
        self._weights.grad *= 0
        self._bias.grad *= 0

    def get_params(self):
        return self._weights, self._bias

class Sgd:
    def __init__(self, model, learning_rate=1e-3):
        self._params = model.get_params()
        self._learning_rate = learning_rate

    def step(self):
        for param in self._params:
            param.data -= self._learning_rate * param.grad

def linear(x):
    return x

def relu(x):
    return torch.relu(x)

def cross_entropy_loss(logits, targets):
    """
    logits should be a Tensor with shape batch_size * num_classes containing
    logit predictions for each class (that will be used as the input to the
    softmax function), and targets should be a 1D Tensor with shape
    num_classes, containing the index of the correct class for each data point
    in the batch
    """
    exp_logits = torch.exp(logits)
    softmax_correct = (
        exp_logits[range(len(targets)), targets]
        / torch.sum(exp_logits, dim=1)
    )
    return -torch.mean(torch.log(softmax_correct))

if __name__ == "__main__":
    mlp = Mlp(28*28, 10, 400, 2, linear, relu)
    optimiser = Sgd(model=mlp, learning_rate=1e-3)

    train_dataset = torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_dataset = torchvision.datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1,
    )

    def test_accuracy(model, test_loader):
        num_test = 0
        num_correct = 0
        for x, target in test_loader:
            y = model.forward(x.reshape(-1, 28*28))
            num_correct += sum(y.argmax(dim=1) == target)
            num_test += len(target)

        return 100 * num_correct / num_test

    print("Test accuracy = %.3f%%" % test_accuracy(mlp, test_loader))
    loss_list = []
    timer = util.Timer()
    for epoch in range(5):
        for i, [x, target] in enumerate(train_loader):
            y = mlp.forward(x.reshape(-1, 28*28))
            loss = cross_entropy_loss(y, target)
            loss.backward()
            optimiser.step()
            mlp.zero_grad()

            loss_list.append(loss.item())
            if i % 100 == 0:
                print(
                    "Epoch %i, batch %3i, loss = %.3f"
                    % (epoch, i, loss.item())
                )

        print("Test accuracy = %.3f%%" % test_accuracy(mlp, test_loader))
        timer.print_time_taken()

    plotting.plot(plotting.Line(loss_list, c="b"))
