import os
import pickle
import __init__
import scripts.course_3_optimisation
import numpy as np
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
    x_i_batch = x_i[:batch_size]
    x_not_i_batch = x_not_i[:batch_size]
    a, b = svm.solve(
        np.block([[x_i_batch], [x_not_i_batch]]),
        np.block([np.ones(batch_size), -np.ones(batch_size)]),
        norm_penalty,
    )
    y_train_i_pred = x_train @ a + b
    y_test_i_pred = x_test @ a + b
    return y_train_i_pred, y_test_i_pred

def predict_all_digits(
    x_train,
    y_train,
    x_test,
    batch_size,
    norm_penalty,
    rng,
):
    test_pred_list = []
    for digit in range(10):
        print(digit)
        _, y_test_i_pred = predict_digit(
            x_train,
            y_train,
            x_test,
            digit,
            batch_size,
            norm_penalty,
            rng,
        )
        test_pred_list.append(y_test_i_pred)

    test_preds = np.argmax(test_pred_list, axis=0)

    return test_preds
