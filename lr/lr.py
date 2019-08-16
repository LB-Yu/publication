import numpy as np


def sigmoid(x):
    """
    sigmoid function
    Args:
        x: scala or numpy ndarray
    """
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    """
    Logistic regression model implement with pure python.
    Same interface with sklearn.

    For easy implementation, this class does not support batch training,
    if you need batch training, see https://github.com/LB-Yu/publication/tree/master/autodiff_py
    """
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, Y):
        """
        Train the model.
        Args:
            X: [n_samples, n_features]
            Y: [n_samples, 1]
        """
        m, n = X.shape
        self.w_ = np.zeros([n, 1])
        self.b_ = 0
        self.cost_ = []

        for i in range(self.max_iter):
            Y_hat = self.predict(X)

            cost = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / m

            if i != 0 and i % 10 == 0:
                print("Step: " + str(i) + ", Cost: " + str(cost))

            self.cost_.append(cost)

            self.w_ -= self.learning_rate * np.dot(X.T, Y_hat - Y) / m
            self.b_ -= self.learning_rate * np.sum(Y_hat - Y) / m

    def predict(self, X):
        """
        Predict the given examples.
        Args:
            X: [n_samples, n_features]
        """
        return sigmoid(np.dot(X, self.w_) + self.b_)

    def score(self, X, Y):
        """
        Score the prediction by the model.
        Args:
            X: [n_samples, n_features]
            Y: [n_samples, 1]

        Returns:
            score = right_samples / total_samples
        """
        Y_hat = self.predict(X).reshape(-1)
        Y_predict = np.array([1 if y >= 0.5 else 0 for y in Y_hat])
        true_num = np.sum(Y_predict == Y.reshape(-1))
        return true_num / len(X)
