import numpy as np
import mnist

rng = np.random.default_rng(0)
x_train, y_train, x_test, y_test = mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
test_preds = mnist.predict_all_digits(
    x_train,
    y_train,
    x_test,
    batch_size=1000,
    norm_penalty=62.647540,
    rng=rng,
)
test_accuracy = np.mean(test_preds == y_test)
print("Test accuracy = %f" % test_accuracy)
