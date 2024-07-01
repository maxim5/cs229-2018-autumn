import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    fcsts = model.predict(x_valid)
    util.evaluate_regression(y_valid, fcsts, "PS1 p05(b)")

    # Plot validation predictions on top of validation set
    fig_path_prefix = "output/p05b_pred_1"
    util.plot_regression_train_and_fcst(x_valid, y_valid, x_valid, fcsts,
                                        save_path=fig_path_prefix)

    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.x = None
        self.y = None
        self.theta = None

    def get_weights(self, x_hat):
        j, n = x_hat.shape
        # 'n' can be overwritten because it's same for both
        m, n = self.x.shape

        # read https://numpy.org/doc/stable/user/basics.broadcasting.html
        x_hat = x_hat.reshape(j, 1, n)

        diffs = self.x - x_hat
        l2_norm = np.linalg.norm(diffs, ord=2, axis=2)
        l2_norm_squared = l2_norm ** 2
        return np.exp(-l2_norm_squared/(2 * self.tau**2))

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        W = self.get_weights(x)
        j, k = W.shape
        W = W.reshape(j, k, 1)
        weighted_X = self.x * W
        weighted_X = np.transpose(weighted_X, axes=(0, 2, 1))

        X = np.copy(self.x)
        Thetas = np.linalg.inv(weighted_X @ X) @ weighted_X @ self.y

        y_hats = np.sum(Thetas * x, 1)
        return y_hats
        # *** END CODE HERE ***
