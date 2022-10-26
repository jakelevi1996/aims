import numpy as np
import __init__
import scripts.course_3_optimisation
import plotting
import util
import mnist

x_train, y_train, x_test, y_test = mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
norm_penalty_list = np.exp(np.linspace(*np.log([0.1, 10]), 20))
train_accuracy = []
test_accuracy = []
rng = np.random.default_rng(0)
batch_size = 1000
total_timer = util.Timer()

for norm_penalty in norm_penalty_list:
    print(norm_penalty, end=", ", flush=True)
    timer = util.Timer()
    y_train_i_pred, y_test_i_pred = mnist.predict_digit(
        x_train,
        y_train,
        x_test,
        digit=5,
        batch_size=batch_size,
        norm_penalty=norm_penalty,
        rng=rng,
    )
    timer.print_time_taken()
    train_accuracy.append(np.mean((y_train_i_pred > 0) == (y_train == 5)))
    test_accuracy.append(np.mean((y_test_i_pred > 0) == (y_test == 5)))

print("Total time taken = %.3f s" % total_timer.time_taken())
best_test_accuracy = max(test_accuracy)
best_norm_penalty = norm_penalty_list[test_accuracy.index(best_test_accuracy)]
print(
    "best_test_accuracy = %f, best_norm_penalty = %f"
    % (best_test_accuracy, best_norm_penalty)
)

plotting.plot(
    plotting.Line(
        norm_penalty_list,
        train_accuracy,
        c="b",
        label="Train accuracy",
    ),
    plotting.Line(
        norm_penalty_list,
        test_accuracy,
        c="r",
        label="Test accuracy",
    ),
    plotting.HVLine(
        h=best_test_accuracy,
        v=best_norm_penalty,
        c="r",
        ls="--",
        label="Best norm_penalty = %.3f"% best_norm_penalty,
    ),
    plot_name=(
        "Train and test accuracy for different norm penalties, "
        "batch_size = %i"
        % batch_size
    ),
    dir_name=scripts.course_3_optimisation.RESULTS_DIR,
    axis_properties=plotting.AxisProperties(
        "Norm penalty",
        "Accuracy",
        log_xscale=True,
    ),
    legend_properties=plotting.LegendProperties(),
)
