import matplotlib.pyplot as plt
import numpy as np
import os


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path=None, correction=0.0):
# def plot(x, y, theta, save_path=None, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2])) * 0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1])) * 0.2
    x1 = np.arange(min(x[:, -2]) - margin1, max(x[:, -2]) + margin1, 0.01)
    x2 = correction - (theta[0] / theta[2] + theta[1] / theta[2] * x1)
    # x2 = -(theta[0] / theta[2]*correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min() - margin1, x[:, -2].max() + margin1)
    plt.ylim(x[:, -1].min() - margin2, x[:, -1].max() + margin2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path is not None:
        plt.savefig(save_path)

def plot_regression(x, y, label_x="x", label_y="y", save_path=""):
    plt.figure()
    plt.scatter(x, y)
    # Add labels and save to disk
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if save_path is not None:
        plt.savefig(save_path)

def plot_regression_train_and_fcst(x_train, y_train, x_test, y_test, label_x="x", label_y="y", save_path=""):
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_test, y_test, 'ro', linewidth=2)
    # Add labels and save to disk
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if save_path is not None:
        plt.savefig(save_path)


def evaluate_classification(fcst, y_test, heading=""):
    print(f"\n\n{heading}")
    # evaluate
    y_test_vec = y_test.reshape(len(y_test), 1)
    fcst_vec = fcst.reshape(len(fcst), 1)

    correct = (y_test_vec == fcst_vec)
    wrong = 1 - correct

    accuracy = np.sum(correct) / len(fcst_vec)
    true_pos = np.sum(correct * fcst_vec)
    true_neg = np.sum(correct * (1 - fcst_vec))
    false_pos = np.sum(wrong * fcst_vec)
    false_neg = np.sum(wrong * (1 - fcst_vec))

    if (true_pos + false_pos) == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)

    if (true_pos + false_neg) == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)

    print(f"y_test_vec, shape: {y_test_vec.shape}, num: {np.sum(y_test_vec)}")
    print(f"fcst_vec, shape: {fcst_vec.shape}, num: {np.sum(fcst_vec)}")
    print(f"correct vec, shape: {correct.shape}, num: {np.sum(correct)}")
    print(f"wrong vec, shape: {wrong.shape}, num: {np.sum(wrong)}")
    print(f"accuracy: {np.round(accuracy, 2)}, "
          f"precision: {np.round(precision, 2)}, "
          f"recall: {np.round(recall, 2)}")


def evaluate_regression(y, y_hat, heading=""):
    print(f"\n\n{heading}")
    m = len(y)
    # evaluate
    y_vec = y.reshape(m, 1)
    y_hat_vec = y_hat.reshape(m, 1)
    mse = (1/m) * np.sum((y_hat_vec - y_vec)**2)
    mse = np.round(mse, 2)
    print(f"MSE = {mse}")
    return mse

def write(data, path):
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    np.savetxt(path, data, fmt="%d", delimiter=",")

def get_fig_prefix(pred_path):
    return pred_path.split(".")[0]
