import torch
import numpy as np

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

    def cuda(self, cuda_device_id=0):
        self._weights = torch.tensor(
            data=self._weights.detach().numpy(),
            device=cuda_device_id,
            requires_grad=True,
        )
        self._bias = torch.tensor(
            data=self._bias.detach().numpy(),
            device=cuda_device_id,
            requires_grad=True,
        )
