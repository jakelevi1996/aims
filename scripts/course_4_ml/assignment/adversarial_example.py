import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import __init__
import nn
import util
import plotting
import scripts.course_4_ml.assignment

if __name__ == "__main__":
    program_timer = util.Timer(name="Full program")

    mlp = scripts.course_4_ml.assignment.get_mnist_model()

    test_dataset = torchvision.datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_iter = iter(test_dataset)
    x_0, t_0 = next(test_iter)
    while t_0 != 0:
        x_0, t_0 = next(test_iter)

    t_adv = 7

    rng = np.random.default_rng(0)
    perturbation = torch.tensor(
        0.01 * rng.normal(size=list(x_0.shape)),
        dtype=torch.float32,
        requires_grad=True,
    )

    loss_list = []
    num_iterations = 100
    max_perturbation = 0.1
    for i in range(num_iterations):
        normalised_perturbation = (
            max_perturbation
            * perturbation
            / torch.max(torch.abs(perturbation))
        )
        x_adv = x_0 + normalised_perturbation
        y_adv = mlp.forward(x_adv)
        loss = nn.loss.cross_entropy_loss(y_adv, [t_adv])
        loss.backward()

        perturbation.data -= 1e-3 * perturbation.grad

        loss_list.append(loss.item())
        print(
            "\rIteration %i, loss = %.3f" % (i, loss.item()),
            end="",
            flush=True,
        )
    print("... Finished optimisation loop")

    plotting.plot(
        plotting.Line(loss_list, c="b"),
        plot_name=(
            "Adversarial loss vs iteration, %i iterations, "
            "maximum pixel perturbation = %.3f"
            % (num_iterations, max_perturbation)
        ),
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=[10, 6],
    )

    y_0 = mlp.forward(x_0).detach().numpy().squeeze()
    y_0_softmax = np.exp(y_0) / np.sum(np.exp(y_0))
    axes[0, 0].set_title(
        "Ground truth = %i, prediction = %i, probability = %.2f"
        % (t_0, np.argmax(y_0_softmax), np.max(y_0_softmax))
    )
    c_0 = "g" if (np.argmax(y_0) == t_0) else "r"
    axes[0, 0].bar(range(10), y_0_softmax, color=c_0)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 1].imshow(x_0.detach().numpy().squeeze())
    axes[0, 1].axis("off")

    y_adv = mlp.forward(x_adv).detach().numpy().squeeze()
    y_adv_softmax = np.exp(y_adv) / np.sum(np.exp(y_adv))
    axes[1, 0].set_title(
        "Maximum pixel perturbation = %.3f, prediction = %i, "
        "probability = %.2f"
        % (
            normalised_perturbation.abs().max().item(),
            np.argmax(y_adv_softmax),
            np.max(y_adv_softmax),
        )
    )
    c_adv = "g" if (np.argmax(y_adv) == t_0) else "r"
    axes[1, 0].bar(range(10), y_adv_softmax, color=c_adv)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 1].imshow(x_adv.detach().numpy().squeeze())
    axes[1, 1].axis("off")

    fig.tight_layout()

    plotting.save_and_close(
        plot_name="Test set predictions with adversarial example",
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        fig=fig,
        verbose=True,
    )

    program_timer.print_time_taken()
