import numpy as np
import util
import os

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    fcsts = model.predict(x_test)
    # print(np.sum((fcsts - y_test)**2))
    model.write(fcsts, pred_path)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        tolerance = 10 ** -5
        change = np.inf
        max_iters = 15
        self.theta = np.zeros(np.shape(x)[1])
        i = 0
        while change > tolerance and i < max_iters:
            grad = self.loss_gradient(self.theta, x, y)
            hess = self.loss_hessian(self.theta, x)
            h_inv = np.linalg.inv(hess)
            self.theta = self.theta - h_inv @ grad
            i += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape ((m, n), m).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        S = self.sigmoid(self.theta, x)
        fcsts = (S >= 0.5) * 1
        return fcsts
        # *** END CODE HERE ***

    def sigmoid(self, theta, X):
        """Evaluate sigmoid based on the given params and X matrix

        Args:
            theta: Input of shape (n, 1)
            X: Input of shape (m, n)

        Returns:
            Outputs of shape (m, 1)
        """
        return 1/(1 + np.exp(-theta @ X.T))

    def loss_gradient(self, theta, X, Y):
        """Evaluate the logistic loss diff

        Args:
            theta: Input of shape (n, 1)
            X: Input of shape (m, n)
            Y: Input of shape (m, 1)
        Returns:
            Outputs of shape (n, 1)
        """
        S = self.sigmoid(theta, X)
        m = len(X[0])
        grad = (1/m)*(np.sum((S - Y) * X.T, axis=1))
        return grad

    def loss_hessian(self, theta, X):
        """Evaluate the logistic loss diff

        Args:
            theta: Input of shape (n, 1)
            X: Input of shape (m, n)
            Y: Input of shape (m, 1)
        Returns:
            Outputs of shape (n, 1)
        """
        S = self.sigmoid(theta, X)
        m = len(X[0])
        return (1/m)*((S*(1 - S) * X.T) @ X)

    def write(self, data, path):
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        np.savetxt(path, data, fmt="%d", delimiter=",")