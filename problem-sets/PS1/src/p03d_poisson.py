import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    model = PoissonRegression()
    model.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    fcsts = model.predict(x_valid)
    util.evaluate_regression(y_valid, fcsts, "PS1 p03(d)")
    fcsts = np.asarray(fcsts, dtype="int")
    util.write(fcsts, pred_path)
    fig_path_prefix = pred_path.split(".")[0]
    util.plot_regression(fcsts, y_valid, "forecast",
                         "true", fig_path_prefix)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, lr=0.003):
        super().__init__()
        self.lr = lr

    def fit(self, x, y, scale_y = "True"):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # calculate the early stopping criteria as 0.01 * prev_loss
        # if the change in loss is less than this value for 3 times in a row,
        # then stop. However, since we may encounter anomalies which can result
        # in a really high loss for a single value. We evaluate this over a
        # batch.
        x = util.add_intercept(x)
        self.scale_param = np.min(y)
        y = y/self.scale_param
        m, n = x.shape
        theta = np.zeros(n)
        for x_i, y_i in zip(x, y):
            theta += self.lr * (y_i - np.exp(theta @ x_i.T)) * x_i
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        x = util.add_intercept(x)
        unscaled_y_hat = np.exp(self.theta @ x.T)
        return unscaled_y_hat * self.scale_param
        # *** END CODE HERE ***
