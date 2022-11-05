import os
import pickle
import numpy as np
import __init__
import nn
import util
import plotting
import scripts.course_4_ml.assignment

if __name__ == "__main__":
    program_timer = util.Timer(name="Full program")
    batch_size_list = [16, 32, 64, 128, 256]
    loss_dict = dict()
    acc_dict = dict()
    for batch_size in batch_size_list:
        mlp = nn.Mlp(
            input_dim=28*28,
            output_dim=10,
            hidden_dim=400,
            num_hidden_layers=2,
            output_act=nn.activation.linear,
            hidden_act=nn.activation.relu,
        )
        mlp.cuda(1)
        train_loader, test_loader = nn.mnist.get_data_loaders(batch_size)
        sgd = nn.optimiser.SgdMomentum(
            model=mlp,
            momentum=0.8,
            learning_rate=1e-3,
        )

        print("Test accuracy = ...", end="", flush=True)
        acc = mlp.get_accuracy(test_loader)
        print("\rTest accuracy = %.3f%%" % acc)
        loss_list = []
        time_list = []
        timer = util.Timer()
        for epoch in range(5):
            print(
                "Epoch %i, batch_size = %i"
                % (epoch, batch_size)
            )
            mlp.train(
                train_loader,
                nn.loss.cross_entropy_loss,
                sgd,
                loss_list,
                time_list,
                timer,
            )

            print("Test accuracy = ...", end="", flush=True)
            acc = mlp.get_accuracy(test_loader)
            print("\rTest accuracy = %.3f%%" % acc)
            timer.print_time_taken()

        loss_dict[batch_size] = [time_list, loss_list]
        acc_dict[batch_size] = acc

    cp = plotting.ColourPicker(len(batch_size_list), cyclic=False)
    line_list = [
        plotting.Line(
            *loss_dict[k],
            color=cp(i),
            label="Batch size = %i, final test accuracy = %.1f%%"
            % (k, acc_dict[k]),
            alpha=0.3,
        )
        for i, k in enumerate(batch_size_list)
    ]
    plotting.plot(
        *line_list,
        plot_name=(
            "MNIST cross entropy loss over 5 epochs vs time, "
            "comparing batch size"
        ),
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
        legend_properties=plotting.LegendProperties(),
    )

    program_timer.print_time_taken()
