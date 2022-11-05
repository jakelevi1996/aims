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
    num_hidden_layers_list = list(range(6))
    loss_dict = dict()
    for num_hidden_layers in num_hidden_layers_list:
        mlp = nn.Mlp(
            input_dim=28*28,
            output_dim=10,
            hidden_dim=400,
            num_hidden_layers=num_hidden_layers,
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
                "Epoch %i, num_hidden_layers = %.3f"
                % (epoch, num_hidden_layers)
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

        loss_dict[num_hidden_layers] = [time_list, loss_list]

    cp = plotting.ColourPicker(len(num_hidden_layers_list), cyclic=False)
    line_list = [
        plotting.Line(*xy, c=cp(i), label="%i hidden layers" % m, alpha=0.3)
        for i, [m, xy] in enumerate(sorted(loss_dict.items()))
    ]
    plotting.plot(
        *line_list,
        plot_name=(
            "MNIST cross entropy loss over 5 epochs vs time, "
            "comparing number of hidden layers"
        ),
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
        legend_properties=plotting.LegendProperties(),
    )

    program_timer.print_time_taken()
