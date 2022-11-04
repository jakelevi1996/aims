import numpy as np
import torch
import __init__
import plotting

w = torch.tensor(np.ones(5), requires_grad=True)
t = torch.tensor(np.arange(5), requires_grad=False)

def loss(w, t):
    return torch.sum(torch.square(w - t))

e = loss(w, t)
e.backward()

error_list = []
w_list = [w.detach().numpy().copy()]
print("Starting optimisation loop...\n")
for i in range(100):
    print("\ri = %i, error = %.5f" % (i, e), end="", flush=True)
    w.grad *= 0
    e = loss(w, t)
    e.backward()
    w.data -= 5e-2 * w.grad
    error_list.append(e)
    w_list.append(w.detach().numpy().copy())

print("\n")

plotting.plot(
    plotting.Line(error_list, c="r"),
    plot_name="Error vs iteration",
)
cp = plotting.ColourPicker(5)
line_list = [
    plotting.Line([w[j] for w in w_list], c=cp(j))
    for j in range(5)
]
plotting.plot(*line_list, plot_name="Weights vs iteration")
