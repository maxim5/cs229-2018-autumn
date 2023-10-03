import numpy as np
import util
import os

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    theta = model.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    fcst = model.predict(x_test)

    # evaluate
    y_test_vec = y_test.reshape(len(y_test), 1)
    correct = (y_test_vec == fcst)
    wrong = 1 - correct
    accuracy = np.sum(correct) / len(fcst)
    true_pos = np.sum(correct * fcst)
    true_neg = np.sum(correct * (1 - fcst))
    false_pos = np.sum(wrong * fcst)
    false_neg = np.sum(wrong * (1 - fcst))

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    print(f"accuracy: {np.round(accuracy, 2)}, "
          f"precision: {np.round(precision, 2)}, "
          f"recall: {np.round(recall, 2)}")

    write(fcst, pred_path)
    fig_path_prefix = pred_path.split(".")[0]
    util.plot(x_train, y_train, theta, save_path=f"{fig_path_prefix}_fig", correction=1.0)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # get m and n
        m, n = x.shape[0], x.shape[1]

        # calculate phi
        phi = np.sum(y) / m

        # calculate mu0 and mu1
        y0, y1 = y == 0, y == 1
        num_0, num_1 = np.sum(y0), np.sum(y1)
        y0, y1 = y0.reshape(len(y0), 1), y1.reshape(len(y1), 1)
        mu0, mu1 = np.sum(x * y0, axis=0) / num_0, np.sum(x * y1,
                                                          axis=0) / num_1

        # calculate sigma
        x_diff = x - (y0 * mu0) - (y1 * mu1)
        sigma = (1 / m) * x_diff.T @ x_diff

        # calculate thetas
        sigma_inv = np.linalg.inv(sigma)
        theta = (mu1 - mu0) @ sigma_inv
        theta0 = (1 / 2) * (mu0 + mu1) @ sigma_inv @ (
                    mu0 - mu1).T - np.log((1 - phi) / phi)

        self.theta = np.insert(theta, 0, theta0)
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # calculate sigmoid
        S = 1 / (1 + np.exp(-(x @ self.theta)))
        return S >= 0.5
        # *** END CODE HERE


def write(data, path):
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    np.savetxt(path, data, fmt="%d", delimiter=",")
