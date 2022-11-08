import os
import sys
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

sys.path.append(CURRENT_DIR)
sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")

MNIST_MODEL_PATH = os.path.join(RESULTS_DIR, "mnist_mlp.pkl")
RNN_MODEL_PATH   = os.path.join(RESULTS_DIR, "shakespeare_rnn.pkl")
LSTM_MODEL_PATH  = os.path.join(RESULTS_DIR, "shakespeare_lstm.pkl")

def get_mnist_model():
    if not os.path.isfile(MNIST_MODEL_PATH):
        print("MNIST model not found, training one now...")
        import save_mlp
        save_mlp.main()

    with open(MNIST_MODEL_PATH, "rb") as f:
        mlp = pickle.load(f)

    return mlp

def get_rnn_model():
    if not os.path.isfile(RNN_MODEL_PATH):
        print("RNN model not found, training one now...")
        import save_rnn
        save_rnn.main()

    with open(RNN_MODEL_PATH, "rb") as f:
        rnn = pickle.load(f)

    return rnn

def get_lstm_model():
    if not os.path.isfile(LSTM_MODEL_PATH):
        print("LSTM model not found, training one now...")
        import save_lstm
        save_lstm.main()

    with open(LSTM_MODEL_PATH, "rb") as f:
        lstm = pickle.load(f)

    return lstm
