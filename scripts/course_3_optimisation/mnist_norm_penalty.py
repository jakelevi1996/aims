import os
import pickle
import numpy as np
import __init__
import scripts.course_3_optimisation
import plotting
import svm

MNIST_PATH = os.path.join(
    scripts.course_3_optimisation.CURRENT_DIR,
    "mnist.pkl",
)

def load_data():
    if os.path.isfile(MNIST_PATH):
        with open(MNIST_PATH, "rb") as f:
            x_train, y_train, x_test, y_test = pickle.load(f)
    else:
        import tensorflow as tf
        mnist_data = tf.keras.datasets.mnist.load_data()
        (x_train, y_train), (x_test, y_test) = mnist_data
        with open(MNIST_PATH, "wb") as f:
            pickle.dump([x_train, y_train, x_test, y_test], f)

    return x_train, y_train, x_test, y_test

def predict_digit(
    x_train,
    y_train,
    x_test,
    digit,
    batch_size,
    norm_penalty,
    rng,
):
    x_i = x_train[y_train == digit]
    x_not_i = x_train[y_train != digit]
    x_i_batch = x_i[
        rng.choice(x_i.shape[0], batch_size, replace=False)
    ]
    x_not_i_batch = x_not_i[
        rng.choice(x_not_i.shape[0], batch_size, replace=False)
    ]
    labels = np.ones(2 * batch_size)
    labels[batch_size:] = -1
    a, b = svm.solve(
        np.block([[x_i_batch], [x_not_i_batch]]),
        labels,
        norm_penalty,
    )
    y_train_i_pred = x_train @ a + b
    y_test_i_pred = x_test @ a + b
    return y_train_i_pred, y_test_i_pred

x_train, y_train, x_test, y_test = load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
norm_penalty_list = np.exp(np.linspace(*np.log([0.1, 10]), 20))
train_accuracy = []
test_accuracy = []
rng = np.random.default_rng(0)
batch_size = 1000

for norm_penalty in norm_penalty_list:
    print(norm_penalty)
    y_train_i_pred, y_test_i_pred = predict_digit(
        x_train,
        y_train,
        x_test,
        digit=5,
        batch_size=batch_size,
        norm_penalty=norm_penalty,
        rng=rng,
    )
    train_accuracy.append(np.mean((y_train_i_pred > 0) == (y_train == 5)))
    test_accuracy.append(np.mean((y_test_i_pred > 0) == (y_test == 5)))

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
