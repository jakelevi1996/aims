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
    num_params_dict = dict()

    rnn = nn.CharRnn(likely_char_list)
    lstm = nn.CharLstm(likely_char_list)
    for model in [lstm, rnn]:
        model_name = type(model).__name__
        num_params = sum(param.numel() for param in model.get_params())
        print("Training model %s, num params = %i" % (model_name, num_params))
        num_params_dict[model_name] = num_params
        sgd = nn.optimiser.SgdMomentum(
            model=model,
            momentum=0.8,
            learning_rate=1e-3,
        )

        time_list, loss_list = model.train(
            data_str=s,
            optimiser=sgd,
            max_num_batches=1000,
            predict_args=["once upon a time"],
        )

        loss_dict[model_name] = [time_list, loss_list]

    rnn_name    = type(rnn ).__name__
    lstm_name   = type(lstm).__name__
    plotting.plot(
        plotting.Line(
            *loss_dict[rnn_name],
            color="b",
            label="RNN, %i parameters" % num_params_dict[rnn_name],
        ),
        plotting.Line(
            *loss_dict[lstm_name],
            color="g",
            label="LSTM, %i parameters" % num_params_dict[lstm_name],
        ),
        plot_name=(
            "Shakespeare RNN vs LSTM, mean cross entropy loss vs time "
            "over 1000 batches of 64 characters each"
        ),
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
        legend_properties=plotting.LegendProperties(),
    )
    pickle_path = os.path.join(
        scripts.course_4_ml.assignment.RESULTS_DIR,
        "Loss dictionary RNN vs LSTM.pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(loss_dict, f)

    program_timer.print_time_taken()
