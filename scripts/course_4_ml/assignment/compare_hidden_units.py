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
    hidden_dim_list = [100, 200, 400, 800, 1600]
    loss_dict = dict()
    acc_dict = dict()
    for hidden_dim in hidden_dim_list:
        mlp = nn.Mlp(
            input_dim=28*28,
            output_dim=10,
            hidden_dim=hidden_dim,
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

        print("Test accuracy = ...", end="", flush=True)
        acc = mlp.get_accuracy(test_loader)
        print("\rTest accuracy = %.3f%%" % acc)
        loss_list = []
        time_list = []
        timer = util.Timer()
        for epoch in range(5):
            print(
                "Epoch %i, hidden_dim = %i"
                % (epoch, hidden_dim)
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

        loss_dict[hidden_dim] = [time_list, loss_list]
        acc_dict[hidden_dim] = acc

    cp = plotting.ColourPicker(len(hidden_dim_list), cyclic=False)
    line_list = [
        plotting.Line(
            *loss_dict[k],
            color=cp(i),
            label="hidden dim = %i, final test accuracy = %.1f%%"
            % (k, acc_dict[k]),
            alpha=0.3,
        )
        for i, k in enumerate(hidden_dim_list)
    ]
    plotting.plot(
        *line_list,
        plot_name=(
            "MNIST cross entropy loss over 5 epochs vs time, "
            "comparing dimension of hidden layers"
        ),
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
        legend_properties=plotting.LegendProperties(),
    )

    program_timer.print_time_taken()
