import os
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_PATH = os.path.join(CURRENT_DIR, "mnist.pkl")

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
