import os
import pickle
import __init__
import nn
import util
import plotting
import scripts.course_4_ml.assignment

if __name__ == "__main__":
    program_timer = util.Timer(name="Full program")
    for num_hidden_units, num_hidden_layers in [[400, 2], [800, 4]]:
        loss_dict = dict()
        for use_gpu in [True, False]:
            mlp = nn.Mlp(
                input_dim=28*28,
                output_dim=10,
                hidden_dim=num_hidden_units,
                num_hidden_layers=num_hidden_layers,
                output_act=nn.activation.linear,
                hidden_act=nn.activation.relu,
            )
            if use_gpu:
                mlp.cuda()
            train_loader, test_loader = nn.mnist.get_data_loaders()
            sgd = nn.optimiser.Sgd(model=mlp, learning_rate=1e-3)

            print("Test accuracy = ...", end="", flush=True)
            acc = mlp.get_accuracy(test_loader)
            print("\rTest accuracy = %.3f%%" % acc)
            loss_list = []
            time_list = []
            timer = util.Timer()
            for epoch in range(5):
                print("Epoch %i" % epoch)
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

            loss_dict[use_gpu] = [time_list, loss_list]

        plotting.plot(
            plotting.Line(*loss_dict[False], c="b", label="CPU"),
            plotting.Line(*loss_dict[True ], c="g", label="GPU"),
            plot_name=(
                "MNIST cross entropy loss over 5 epochs vs time, CPU vs GPU, "
                "%i hidden layers, %i hidden units"
                % (num_hidden_layers, num_hidden_units)
            ),
            dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
            axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
            legend_properties=plotting.LegendProperties(),
        )
        pickle_path = os.path.join(
            scripts.course_4_ml.assignment.RESULTS_DIR,
            "Loss dictionary %i layers %i units.pkl"
            % (num_hidden_layers, num_hidden_units)
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(loss_dict, f)

    program_timer.print_time_taken()
