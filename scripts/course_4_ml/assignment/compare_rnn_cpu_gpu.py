import os
import pickle
import __init__
import nn
import util
import plotting
import scripts.course_4_ml.assignment

if __name__ == "__main__":
    program_timer = util.Timer(name="Full program")

    s = nn.data.shakespeare.get_data().lower()
    likely_char_list = nn.data.shakespeare.get_likely_chars(s)
    loss_dict = dict()

    for use_gpu in [False, True]:
        rnn = nn.CharRnn(likely_char_list)
        if use_gpu:
            rnn.cuda()
        train_loader, test_loader = nn.mnist.get_data_loaders()
        sgd = nn.optimiser.SgdMomentum(
            model=rnn,
            momentum=0.8,
            learning_rate=1e-3,
        )

        time_list, loss_list = rnn.train(
            data_str=s,
            optimiser=sgd,
            max_num_batches=1000,
            predict_args=["once upon a time"],
        )

        loss_dict[use_gpu] = [time_list, loss_list]

    plotting.plot(
        plotting.Line(*loss_dict[False], c="b", label="CPU"),
        plotting.Line(*loss_dict[True ], c="g", label="GPU"),
        plot_name=(
            "Shakespeare RNN mean cross entropy loss vs time "
            "over 1000 batches of 64 characters each, CPU vs GPU"
        ),
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
        legend_properties=plotting.LegendProperties(),
    )
    pickle_path = os.path.join(
        scripts.course_4_ml.assignment.RESULTS_DIR,
        "Loss dictionary RNN.pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(loss_dict, f)

    program_timer.print_time_taken()
