import torch
import numpy as np
import util
import nn

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
        self._input_dim = input_dim
        self._cuda_device_id = None
        if rng is None:
            rng = np.random.default_rng(0)
        if num_hidden_layers > 0:
            self._layers = [
                nn.layer.Layer(
                    input_dim,
                    hidden_dim,
                    hidden_act,
                    bias_std,
                    rng,
                )
            ]
            for _ in range(num_hidden_layers - 1):
                self._layers.append(
                    nn.layer.Layer(
                        hidden_dim,
                        hidden_dim,
                        hidden_act,
                        bias_std,
                        rng,
                    )
                )
            self._layers.append(
                nn.layer.Layer(
                    hidden_dim,
                    output_dim,
                    output_act,
                    bias_std,
                    rng,
                )
            )
        else:
            self._layers = [
                nn.layer.Layer(
                    input_dim,
                    output_dim,
                    output_act,
                    bias_std,
                    rng,
                )
            ]

    def forward(self, x):
        batch_size = x.shape[0]
        layer_output = x.reshape(batch_size, self._input_dim).T
        for layer in self._layers:
            layer_output = layer.forward(layer_output)
        return layer_output.T

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

    def cuda(self, cuda_device_id=0):
        self._cuda_device_id = cuda_device_id
        for layer in self._layers:
            layer.cuda(cuda_device_id)

    def get_params(self):
        param_list = [
            param
            for layer in self._layers
            for param in layer.get_params()
        ]
        return param_list

    def train(
        self,
        train_loader,
        loss_func,
        optimiser,
        loss_list=None,
        time_list=None,
        timer=None,
        num_epochs=1,
        print_every=100,
    ):
        if loss_list is None:
            loss_list = []
        if time_list is None:
            time_list = []
        if timer is None:
            timer = util.Timer()

        for epoch in range(num_epochs):
            for i, [x, target] in enumerate(train_loader):
                if self._cuda_device_id is not None:
                    x = x.cuda(self._cuda_device_id)
                    target = target.cuda(self._cuda_device_id)
                y = self.forward(x)
                loss_tensor = loss_func(y, target)
                loss_tensor.backward()
                optimiser.step()
                self.zero_grad()

                loss = loss_tensor.item()
                loss_list.append(loss)
                time_list.append(timer.time_taken())
                if i % print_every == 0:
                    print("Batch %4i | loss = %.3f" % (i, loss))

        return time_list, loss_list

    def get_accuracy(self, data_loader):
        num_test = 0
        num_correct = 0
        for x, target in data_loader:
            if self._cuda_device_id is not None:
                x = x.cuda(self._cuda_device_id)
                target = target.cuda(self._cuda_device_id)
            y = self.forward(x)
            num_correct += sum(y.argmax(dim=1) == target)
            num_test += len(target)

        return 100 * num_correct / num_test
