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
    activation_list = [
        nn.activation.linear,
        nn.activation.relu,
        nn.activation.sigmoid,
        nn.activation.gaussian,
        nn.activation.cauchy,
    ]
    loss_dict = dict()
    acc_dict = dict()
    for activation in activation_list:
        mlp = nn.Mlp(
            input_dim=28*28,
            output_dim=10,
            hidden_dim=400,
            num_hidden_layers=2,
            output_act=nn.activation.linear,
            hidden_act=activation,
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
                "Epoch %i, activation = %s"
                % (epoch, activation.__name__)
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

        loss_dict[activation] = [time_list, loss_list]
        acc_dict[activation] = acc

    cp = plotting.ColourPicker(len(activation_list), cyclic=True)
    line_list = [
        plotting.Line(
            *loss_dict[k],
            color=cp(i),
            label="Hidden activation = %s, final test accuracy = %.1f%%"
            % (k.__name__, acc_dict[k]),
            alpha=0.3,
        )
        for i, k in enumerate(activation_list)
    ]
    plotting.plot(
        *line_list,
        plot_name=(
            "MNIST cross entropy loss over 5 epochs vs time,\n"
            "comparing hidden activation functions"
        ),
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
        legend_properties=plotting.LegendProperties(),
    )

    program_timer.print_time_taken()
