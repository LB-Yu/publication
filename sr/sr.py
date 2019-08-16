import numpy as np


def softmax(x):
    """
    Softmax regression for a vector or matrix.
    Args:
        x: [n_examples, n_classes]
    Returns: values after softmax.
    """
    # b = x - np.max(x, axis=1, keepdims=True)
    expb = np.exp(x)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


class SoftmaxRegression:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        """
        Train the model.
        Args:
            X: [n_samples, n_features]
            Y: [n_samples, n_classes]
        """
        m, n = X.shape
        _, K = Y.shape
        self.w_ = np.zeros([n, K])
        self.b_ = np.zeros([1, K])
        self.cost_ = []

        for i in range(self.max_iter):
            Y_hat = self.predict(X)

            cost = -np.sum(Y * np.log(Y_hat)) / m

            if i != 0 and i % 10 == 0:
                print("Step: " + str(i) + ", Cost: " + str(cost))

            self.cost_.append(cost)

            self.w_ -= self.learning_rate * np.dot(X.T, Y_hat - Y) / m
            self.b_ -= self.learning_rate * np.sum(Y_hat - Y, axis=0) / m

    def predict(self, X):
        """
        Predict the given examples.
        Args:
            X: [n_samples, n_features]
        """
        z = np.dot(X, self.w_)
        return softmax(np.dot(X, self.w_) + self.b_)

    def score(self, X, Y):
        Y_hat = self.predict(X)
        Y_hat = np.argmax(Y_hat, axis=1)
        Y = np.argmax(Y, axis=1)
        true_num = np.sum(Y_hat == Y)
        return true_num / len(X)
