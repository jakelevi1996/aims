import os
import pickle
import matplotlib.pyplot as plt
import __init__
import nn
import plotting
import util
import scripts.course_4_ml.assignment

def main():
    s = nn.data.shakespeare.get_data().lower()
    likely_char_list = nn.data.shakespeare.get_likely_chars(s)
    rnn = nn.CharRnn(likely_char_list)
    sgd = nn.optimiser.SgdMomentum(
        model=rnn,
        momentum=0.8,
        learning_rate=1e-4,
    )

    time_list, loss_list = rnn.train(
        data_str=s,
        optimiser=sgd,
        max_num_batches=len(s),
        max_num_seconds=(30 * 60),
        predict_args=["once upon a time"],
        predict_every=500,
    )
    rnn.predict("once upon a time")

    rnn_model_path = scripts.course_4_ml.assignment.RNN_MODEL_PATH
    print("Saving model in \"%s\"" % rnn_model_path)
    with open(rnn_model_path, "wb") as f:
        pickle.dump(rnn, f)

    results_dir = scripts.course_4_ml.assignment.RESULTS_DIR
    learning_curves_path = os.path.join(results_dir, "rnn_learning_curve.pkl")
    with open(learning_curves_path, "wb") as f:
        pickle.dump([time_list, loss_list], f)

    plotting.plot(
        plotting.Line(time_list, loss_list, color="b"),
        plot_name="Shakespeare RNN training curve",
        dir_name=scripts.course_4_ml.assignment.RESULTS_DIR,
        axis_properties=plotting.AxisProperties("Time (s)", "Loss"),
    )

if __name__ == "__main__":
    with util.Timer(name="Full program"):
        main()
