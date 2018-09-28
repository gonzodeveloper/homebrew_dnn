import numpy as np
import random
import collections


class NeuralNet:
    """
    Simple Dense Neural Network. Tuned to run with any given number of layers with arbitrary sizes.
    Currently supports tanh as an activation function for hidden layers and softmax for output.
    """
    class Layer:
        """
        Subclass for creating input, output and hidden layers. Used by Neural Net constructor
        """
        def __init__(self, func, n_in, n_out):
            # Initial weights centered around 0, proportional to inputs
            self.weights = ((2 * np.random.random((n_in, n_out)) - 1) / n_in)
            self.bias = np.zeros(n_out)
            self.func = func

        def activation(self, x):
            if self.func == "tanh":
                return np.tanh(x)
            if self.func == "softmax":
                return np.exp(x) / np.sum(np.exp(x), axis = 1).reshape((-1,1))

        def deriv(self, y):
            if self.func == "tanh":
                return 1 - np.tanh(y) ** 2
            if self.func == 'softmax':
                s = y.reshape(-1, 1)
                return np.diagflat(s) - np.dot(s, s.T)

    def __init__(self, layers):
        """
        Constructor for NeuralNet
        :param layers: list of tuples specifying layers activation functions and sizes.
        e.g., [("tanh", (64,32)), ("tanh", (32, 16)), ("softmax", (16, 8))] creates a NN with 2 hidden tanh
        layers with 32 and 16 nodes, as well a softmax output layer with 8 nodes.
        """
        self.layers = []
        for items in layers:
            func, size = items
            n_in, n_out = size
            self.layers.append(self.Layer(func, n_in, n_out))

    def train(self, data, labels, epochs=100, batch_size=50, learning_rate=.01):
        """
        Trains the model using Stochastic Gradient Decent of mini-batched data
        :param data: training data, ndarray
        :param labels: training labels, ndarray
        :param epochs: number of training epochs, int
        :param batch_size: mini-batch size, int
        :param learning_rate: learning rate, float
        :return: lists of validation costs and predictions scores averaged by epoch
        """
        training_data = list(zip(data, labels))
        validation_costs = []
        prediction_scores = []

        for i in range(epochs):
            NeuralNet.print_progress_bar(i + 1, epochs)

            # Shuffle and batch
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size]
                            for k in range(0, len(training_data), batch_size)]

            batch_costs = []
            batch_scores  = []
            for batches in mini_batches:
                data = np.array([x[0] for x in batches])
                targets = np.array([x[1] for x in batches])

                # Run batch through NN and cache the activation results
                activations, predictions = self._feedforward_and_predict(data)

                # Get training score for mini batch
                accuracy =  self.score(targets, predictions)

                # Update weights through SGD and get cross entropy loss
                cost = self._backpropagate(activations, targets, learning_rate)
                batch_costs.append(cost)
                batch_scores.append(accuracy)

            validation_costs.append(np.mean(batch_costs))
            prediction_scores.append(np.mean(batch_scores))

        return validation_costs, prediction_scores

    def predict(self, data):
        """
        Run test data through trained NeuralNet to get predictions (raw output vectors)
        :param data: testing data, ndarray
        :return: predictions, ndarray
        """
        activations = self._feed_forward(data)
        predictions = activations[-1]
        return predictions

    def _feedforward_and_predict(self, data):
        """
        Same as previous method, used for tracking accuracy during training
        :param data: training data
        :return: activation matrices for layers, precidtions, ndarray
        """
        activations = self._feed_forward(data)
        predictions = activations[-1]
        return activations, predictions

    def _feed_forward(self, x):
        """
        Feed data through NeuralNet to get activation matrices. Used in training.
        :param x: data, ndarray
        :return: layer activation matrices, ndarray
        """
        activations = [x]
        for l in self.layers:
            x = l.activation(np.dot(x, l.weights) + l.bias)
            activations.append(x)
        return activations

    def _backpropagate(self, activations, t, learning_rate):
        """
        Use gradient decent to update weights with a given set of activations and targets.
        Also calculates validation loss (cross entropy loss)
        :param activations: matrices of activations from last feed forward
        :param t: output targets
        :param learning_rate: rate by which we adjust weights and biases
        :return: validation loss, float.
        """
        cost = 0
        gradient = None
        deltas = collections.deque()
        m = t.shape[0]
        layers_nums = range(len(self.layers))

        # Iterate through layers backwards, get outputs as y and activations of previous layer as a
        for i in reversed(layers_nums):
            y = activations[i+1]
            a = activations[i]

            # If we are on our softmax output layer dL/dz = y -t
            if gradient is None:
                cost = NeuralNet.cross_entropy_loss(y, t, m)
                gradient = (y - t)

            # If we are on a hidden layer first calculate dL/dh with weights of previous
            # dL/dz = dL/dh * d/dy    (Using derivative of activation function for current hidden layer
            else:
                dLdh = gradient.dot(self.layers[i+1].weights.T)
                gradient = np.multiply(dLdh, self.layers[i].deriv(y))

            # Calculate dL/dw and dL/db, cache these gradients
            weight_grad = 1/m * (a.T).dot(gradient)
            bias_grad = 1/m * np.sum(gradient, axis=0)

            deltas.appendleft((weight_grad, bias_grad))

        # Update weights and biases for each layer
        for j in reversed(layers_nums):
            weight_grad, bias_grad = deltas[j]

            self.layers[j].weights -=  weight_grad * learning_rate
            self.layers[j].bias -= bias_grad * learning_rate

        return cost

    def score(self, truth, predictions):
        """
        Get accuracy score for a set of predictions and their targets. To get the multiclass predictions
        we take the argmax for each of the output and prediction vectors
        :param truth: matrix of target values (one hot encoded)
        :param predictions: matrix of prediction vectors
        :return: accuracy value a,  0 <= a <= 1
        """

        if len(truth) != len(predictions):
            raise ValueError("Array sizes do not match")

        correct = 0
        total = len(truth)

        x_digit = [np.argmax(i) for i in truth]
        y_digit = [np.argmax(j) for j in predictions]

        for x, y in zip(x_digit, y_digit):
            if x == y:
                correct += 1

        return correct / total

    @staticmethod
    def cross_entropy_loss(y, t, m):
        return - np.multiply(t, np.log(y)).sum() / m

    @staticmethod
    def print_progress_bar(iteration, total, length=50, fill='█'):

        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print('\r Epoch %s  of |%s| : %s Total' % (iteration, bar, total), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()
