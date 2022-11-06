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
    mlp = nn.Mlp(
        input_dim=28*28,
        output_dim=10,
        hidden_dim=400,
        num_hidden_layers=2,
        output_act=nn.activation.linear,
        hidden_act=nn.activation.relu,
    )
    mlp.cuda(1)
    train_loader, test_loader = nn.mnist.get_data_loaders()
    sgd = nn.optimiser.SgdMomentum(
        model=mlp,
        momentum=0.8,
        learning_rate=1e-3,
    )

    loss_list = []
    time_list = []
    timer = util.Timer()
    for epoch in range(5):
    # for epoch in range(1):
        print("Epoch %i" % epoch)
        mlp.train(
            train_loader,
            nn.loss.cross_entropy_loss,
            sgd,
            loss_list,
            time_list,
            timer,
        )

        timer.print_time_taken()

    fig, axes = plt.subplots(10, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=[10, 20])
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

        y = mlp.forward(x.cuda(1)).cpu().detach().numpy().squeeze()
        axes[i, 0].set_title(
            "Ground truth = %i, prediction = %i"
            % (i, np.argmax(y))
        )
        axes[i, 0].bar(range(10), np.exp(y) / np.sum(np.exp(y)), color="b")
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
    # line_list = [
    #     plotting.Line(
    #         *loss_dict[k],
    #         color=cp(i),
    #         label="Batch size = %i, final test accuracy = %.1f%%"
    #         % (k, acc_dict[k]),
    #         alpha=0.3,
    #     )
    #     for i, k in enumerate(batch_size_list)
    # ]
    # plotting.plot(
    #     *line_list,
    #     plot_name=(
    #         "MNIST cross entropy loss over 5 epochs vs time, "
    #         "comparing batch size"
    #     ),
    #     dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
    #     axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
    #     legend_properties=plotting.LegendProperties(),
    # )

    program_timer.print_time_taken()
