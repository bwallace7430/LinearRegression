
import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    # Begin with weights and bias at 0. In the math of linear regression, the cost function finds the average error of yPrediction given yActual.
    # The cost function can be defined as: (1/(2*numSamples))*sum((yPrediction-yActual)**2)
    # Following the process of gradient descent, repeatedly update the weights and bias by doing the following:
    # 1. Multiply the learning rate by the derivative of the cost function with respect to the weights. Update weights by subtracting this from the current weights.
    # 2. Multiply the learning rate by the derivative of the cost function with respect to the bias. Update bias by subtracting this from the current bias.
    # If a good learning rate is chosen, this will find a choice of weights and a bias that minimize the cost function.
    # This will result in the least average error in yPrediction, and will give the line of best fit through the data points.

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradual descent
        for _ in range(self.n_iterations):
            y_prediction = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_prediction-y))
            db = (1/n_samples) * np.sum(y_prediction-y)

            # update weights and biases
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    # Use the line found in fit() to produce yPrediction given a piece of data.
    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction
