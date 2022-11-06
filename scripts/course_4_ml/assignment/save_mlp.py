import pickle
import matplotlib.pyplot as plt
import __init__
import nn
import util
import scripts.course_4_ml.assignment

def main():
    program_timer = util.Timer(name="Full program")
    mlp = nn.Mlp(
        input_dim=28*28,
        output_dim=10,
        hidden_dim=400,
        num_hidden_layers=2,
        output_act=nn.activation.linear,
        hidden_act=nn.activation.relu,
    )
    train_loader, test_loader = nn.mnist.get_data_loaders()
    sgd = nn.optimiser.SgdMomentum(
        model=mlp,
        momentum=0.8,
        learning_rate=1e-3,
    )

    timer = util.Timer()
    for epoch in range(5):
        print("Epoch %i" % epoch)
        mlp.train(train_loader, nn.loss.cross_entropy_loss, sgd)

        timer.print_time_taken()

    mnist_model_path = scripts.course_4_ml.assignment.MNIST_MODEL_PATH
    print("Saving model in \"%s\"" % mnist_model_path)
    with open(mnist_model_path, "wb") as f:
        pickle.dump(mlp, f)

    program_timer.print_time_taken()

if __name__ == "__main__":
    main()
