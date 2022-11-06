import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import __init__
import nn
import util
import plotting
import scripts.course_4_ml.assignment

if __name__ == "__main__":
    program_timer = util.Timer(name="Full program")

    mlp = scripts.course_4_ml.assignment.get_mnist_model()

    fig, axes = plt.subplots(
        nrows=10,
        ncols=2,
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=[10, 20],
    )
    test_dataset = torchvision.datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    for i in range(10):
        test_iter = iter(test_dataset)
        x, t = next(test_iter)
        while t != i:
            x, t = next(test_iter)

        y = mlp.forward(x).detach().numpy().squeeze()
        y_softmax = np.exp(y) / np.sum(np.exp(y))
        axes[i, 0].set_title(
            "Ground truth = %i, prediction = %i, confidence = %.2f%%"
            % (i, np.argmax(y_softmax), 100 * np.max(y_softmax))
        )
        c = "g" if (np.argmax(y_softmax) == i) else "r"
        axes[i, 0].bar(range(10), y_softmax, color=c)
        axes[i, 0].set_ylim([0, 1])
        axes[i, 1].imshow(x.squeeze())
        axes[i, 1].axis("off")

    fig.tight_layout()

    plotting.save_and_close(
        plot_name="Test set predictions",
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        fig=fig,
        verbose=True,
    )

    program_timer.print_time_taken()
