from mnist import MNIST
from nerual_net import NeuralNet
import matplotlib.pyplot as plt
import numpy as np


def one_hot_encode(data):
    zeros = np.zeros((len(data), 10))
    zeros[np.arange(len(zeros)), data] += 1
    return zeros


if __name__ == "__main__":
    print("Loading Data...")
    mndata = MNIST('data')

    # Load data, normalize values and encode labels
    train_img, train_lab = mndata.load_training()
    train_img = np.array(train_img) / 255
    train_lab = one_hot_encode(train_lab)

    test_img, test_lab = mndata.load_testing()
    test_img = np.array(test_img) / 255
    test_lab = one_hot_encode(test_lab)

    # Build our DNN, train and predict
    model = NeuralNet([("tanh", (784, 128)), ("tanh", (128, 32)), ("softmax", (32, 10))])

    print("Training Network...")
    validation_costs, training_scores = model.train(train_img, train_lab)

    print("Making predictions...")
    predictions = model.predict(test_img)
    accuracy = model.score(predictions, test_lab)

    # Plot Results
    fig, axes = plt.subplots(1,2)
    axes[0, 0].set_title("Validation costs")
    axes[0, 1].set_title("Training Accuracy")

    axes[0, 1].text(x=0.9, y=0.1, s="Test Accuracy : {}%".format(accuracy * 100))

    for x, y in enumerate(validation_costs):
        axes[0, 0].scatter(x, y, color='c', alpha=0.9)

    for x, y in enumerate(training_scores):
        axes[0, 1].scatter(x, y, color='b', alpha=0.8)

    plt.tight_layout()
    plt.show()